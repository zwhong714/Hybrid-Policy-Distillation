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

import logging
import os

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelHPDActor(DataParallelPPOActor):
    """Recipe-local actor that keeps HPD-specific auxiliary logits out of core verl APIs."""

    def _require_hpd_ready(self) -> None:
        if not self.use_remove_padding:
            raise NotImplementedError("HPD currently requires actor_rollout_ref.model.use_remove_padding=True.")

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False):
        self._require_hpd_ready()
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "greedy_pos" in data.batch.keys():
            select_keys.append("greedy_pos")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        greedy_log_probs_lst = []
        greedy_pos_lst = []

        recompute_with_given_greedy = "greedy_pos" in data.batch.keys()
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if not recompute_with_given_greedy:
                model_inputs["__hpd_enable__"] = True

            with torch.no_grad():
                entropy, log_probs, greedy_log_probs, greedy_pos = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )

            log_probs_lst.append(log_probs)
            greedy_log_probs_lst.append(greedy_log_probs)
            if not recompute_with_given_greedy:
                greedy_pos_lst.append(greedy_pos)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        greedy_log_probs = torch.concat(greedy_log_probs_lst, dim=0)
        greedy_pos = None if recompute_with_given_greedy else torch.concat(greedy_pos_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            greedy_log_probs = restore_dynamic_batch(greedy_log_probs, batch_idx_list)
            if greedy_pos is not None:
                greedy_pos = restore_dynamic_batch(greedy_pos, batch_idx_list)
            if entropys is not None:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys, greedy_log_probs, greedy_pos

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._require_hpd_ready()
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "old_greedy_log_probs",
            "greedy_pos",
            "ref_greedy_log_prob",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob, greedy_log_prob, _ = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    mismatch_mask = model_inputs["input_ids"] != model_inputs["greedy_pos"]
                    ref_greedy_log_prob = model_inputs["ref_greedy_log_prob"]
                    old_greedy_log_prob = model_inputs["old_greedy_log_probs"]
                    reverse_indicator = (ref_greedy_log_prob - old_greedy_log_prob) * torch.exp(old_greedy_log_prob)
                    reverse_advantages = reverse_indicator * mismatch_mask[:, -reverse_indicator.shape[1] :]
                    reverse_negative = reverse_advantages < 0
                    reverse_advantages = torch.where(
                        reverse_negative,
                        reverse_advantages,
                        torch.zeros_like(reverse_advantages),
                    )

                    ref_log_prob = model_inputs["ref_log_prob"]
                    old_log_prob = model_inputs["old_log_probs"]
                    forward_indicator = (ref_log_prob - old_log_prob) * torch.exp(old_log_prob)
                    forward_negative = forward_indicator < 0
                    forward_advantages = torch.where(
                        forward_negative,
                        forward_indicator,
                        torch.exp(ref_log_prob) + forward_indicator,
                    )
                    overlap_mask = reverse_negative & (~forward_negative)
                    forward_advantages = torch.where(
                        overlap_mask,
                        forward_advantages + torch.exp(ref_log_prob),
                        forward_advantages,
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                        old_greedy_log_prob = greedy_log_prob.detach()

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss_reverse, pg_clipfrac_reverse, ppo_kl_reverse, pg_clipfrac_lower_reverse = policy_loss_fn(
                        old_log_prob=old_greedy_log_prob,
                        log_prob=greedy_log_prob,
                        advantages=reverse_advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )
                    pg_loss_forward, pg_clipfrac_forward, ppo_kl_forward, pg_clipfrac_lower_forward = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=forward_advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )
                    pg_loss = pg_loss_reverse + pg_loss_forward
                    pg_clipfrac = 0.5 * (pg_clipfrac_reverse + pg_clipfrac_forward)
                    ppo_kl = 0.5 * (ppo_kl_reverse + ppo_kl_forward)
                    pg_clipfrac_lower = 0.5 * (pg_clipfrac_lower_reverse + pg_clipfrac_lower_forward)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_loss_forward": pg_loss_forward.detach().item() * loss_scale_factor,
                            "actor/pg_loss_reverse": pg_loss_reverse.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
