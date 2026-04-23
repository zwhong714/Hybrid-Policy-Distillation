# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
from verl.workers.fsdp_workers import ActorRolloutRefWorker as BaseActorRolloutRefWorker
from verl.workers.fsdp_workers import logger as fsdp_logger

from .dp_actor import DataParallelHPDActor

__all__ = ["ActorRolloutRefWorker"]


class ActorRolloutRefWorker(BaseActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelHPDActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_ref:
            self.ref_policy = DataParallelHPDActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys, greedy_log_probs, greedy_pos = self.actor.compute_log_prob(
                    data=data,
                    calculate_entropy=True,
                )
            output = DataProto.from_dict(
                tensors={
                    "old_log_probs": output,
                    "entropys": entropys,
                    "old_greedy_log_probs": greedy_log_probs,
                    "greedy_pos": greedy_pos,
                },
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=fsdp_logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            return DataProto.from_dict(
                tensors={
                    "ref_log_prob": data.batch["old_log_probs"],
                    "ref_greedy_log_prob": data.batch["old_greedy_log_probs"],
                }
            )

        assert self._is_ref
        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            output, _, greedy_log_probs, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(
                tensors={"ref_log_prob": output, "ref_greedy_log_prob": greedy_log_probs}
            )

        output = output.to("cpu")

        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)

        return output
