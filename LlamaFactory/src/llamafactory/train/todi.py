# Copyright 2025 the LlamaFactory team.
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

"""Token-wise distillation loss from ToDi."""

import torch
import torch.nn.functional as F

from .akl import _align_shared_vocab, _masked_mean, _sanitize_logits


def compute_todi_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the ToDi loss with detached token-wise FKL/RKL mixing weights."""
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    student_logits, teacher_logits, labels = _align_shared_vocab(student_logits, teacher_logits, labels, ignore_index)

    valid_mask = labels.ne(ignore_index)
    if not valid_mask.any():
        zero = student_logits.new_zeros(())
        return zero, {
            "todi_loss": 0.0,
            "todi_fkl_loss": 0.0,
            "todi_rkl_loss": 0.0,
            "todi_alpha_mean": 0.0,
        }

    student_logits, student_inf_mask = _sanitize_logits(student_logits)
    teacher_logits, teacher_inf_mask = _sanitize_logits(teacher_logits)
    inf_mask = student_inf_mask | teacher_inf_mask

    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    log_ratio = teacher_log_probs - student_log_probs
    alpha = torch.sigmoid(beta * log_ratio).detach()
    fkl_terms = (teacher_probs * (teacher_log_probs - student_log_probs)).masked_fill(inf_mask, 0.0)
    rkl_terms = (student_probs * (student_log_probs - teacher_log_probs)).masked_fill(inf_mask, 0.0)
    todi_terms = (alpha * fkl_terms + (1.0 - alpha) * rkl_terms).masked_fill(inf_mask, 0.0)

    fkl_loss = fkl_terms.sum(dim=-1)
    rkl_loss = rkl_terms.sum(dim=-1)
    todi_loss = todi_terms.sum(dim=-1)
    normalizer = valid_mask.sum() + 1e-8
    loss = (todi_loss * valid_mask.to(todi_loss.dtype)).sum() / normalizer

    metrics = {
        "todi_loss": loss.detach().item(),
        "todi_fkl_loss": ((fkl_loss * valid_mask.to(fkl_loss.dtype)).sum() / normalizer).detach().item(),
        "todi_rkl_loss": ((rkl_loss * valid_mask.to(rkl_loss.dtype)).sum() / normalizer).detach().item(),
        "todi_alpha_mean": _masked_mean(alpha.mean(dim=-1), valid_mask).item(),
    }
    return loss, metrics
