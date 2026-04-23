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

import torch
import torch.nn.functional as F

from .akl import _align_shared_vocab, _sanitize_logits


def _masked_ratio(mask: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return 0.0

    return ((mask & valid_mask).float().sum() / valid_count.float()).item() * 100.0


def _masked_stats(prefix: str, tensor: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().masked_select(valid_mask)
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_neg_pct": 0.0,
        }

    return {
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_min": values.min().item(),
        f"{prefix}_max": values.max().item(),
        f"{prefix}_neg_pct": (values < 0).float().mean().item() * 100.0,
    }


def compute_hpd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    sample_in_fp32: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    sample_logits, teacher_logits, labels = _align_shared_vocab(sample_logits, teacher_logits, labels, ignore_index)
    student_logits, _ = _sanitize_logits(sample_logits)
    teacher_logits, _ = _sanitize_logits(teacher_logits)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    if labels.dim() == student_log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    labels = torch.clamp(labels, min=0)
    student_nll_loss = student_log_probs.gather(dim=-1, index=labels)

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_nll_loss = teacher_log_probs.gather(dim=-1, index=labels)

    k1_gt_raw = (teacher_nll_loss - student_nll_loss) * torch.exp(student_nll_loss)
    mask1 = k1_gt_raw > 0

    adv1 = torch.zeros_like(k1_gt_raw)
    adv1[mask1] = torch.exp(teacher_nll_loss)[mask1] + k1_gt_raw[mask1]
    adv1[~mask1] = k1_gt_raw[~mask1]

    student_nll_loss.masked_fill_(padding_mask, 0.0)

    if sample_in_fp32:
        student_probs = torch.softmax(student_logits, dim=-1)
    else:
        sample_inf_mask = torch.isinf(sample_logits)
        if sample_inf_mask.any():
            sample_logits = sample_logits.masked_fill(sample_inf_mask, 0.0)

        student_probs = torch.softmax(sample_logits, dim=-1)

    batch_size, seq_len, vocab_size = student_probs.shape
    sampled_labels = torch.multinomial(student_probs.view(-1, vocab_size), num_samples=1)
    sampled_labels = sampled_labels.view(batch_size, seq_len).unsqueeze(-1)

    sampled_student_nll_loss = student_log_probs.gather(dim=-1, index=sampled_labels)
    sampled_teacher_nll_loss = teacher_log_probs.gather(dim=-1, index=sampled_labels)

    k1_sample = (sampled_teacher_nll_loss - sampled_student_nll_loss) * torch.exp(sampled_student_nll_loss)

    mask2 = k1_sample < 0
    adv2 = torch.zeros_like(k1_sample)
    adv2[mask2] = k1_sample[mask2]

    mask3 = mask1 & mask2
    adv1[mask3] += torch.exp(teacher_nll_loss)[mask3]
    sampled_student_nll_loss.masked_fill_(padding_mask, 0.0)

    active_mask = ~padding_mask
    sampled_diff_mask = active_mask & sampled_labels.ne(labels)
    adv2_active_mask = sampled_diff_mask & mask2
    mask3_active_mask = active_mask & mask3

    adv1 = adv1.detach()
    adv2 = adv2.detach()

    ce_loss = -student_nll_loss
    
    hpd_loss = -student_nll_loss * adv1 - adv2 * sampled_student_nll_loss * (labels != sampled_labels)

    num_active_elements = active_mask.sum() + 1e-8
    ce_loss_mean = ce_loss.sum() / num_active_elements
    hpd_loss_mean = hpd_loss.sum() / num_active_elements
    loss = hpd_loss.sum() / num_active_elements
    neg_student_nll_loss = (-student_nll_loss.detach()).sum() / num_active_elements

    metrics = {
        "neg_student_nll_loss": neg_student_nll_loss.item(),
        "ce_loss": ce_loss_mean.detach().item(),
        "hpd_loss": hpd_loss_mean.detach().item(),
        "sampled_diff_pct": _masked_ratio(sampled_labels.ne(labels), active_mask),
        "adv1_neg_pct": _masked_ratio(adv1 < 0, active_mask),
        "adv2_active_pct": _masked_ratio(adv2_active_mask, active_mask),
        "mask3_pct": _masked_ratio(mask3_active_mask, active_mask),
    }
    metrics.update(_masked_stats("k1_gt_raw", k1_gt_raw, active_mask))
    metrics.update(_masked_stats("k1_sample", k1_sample, sampled_diff_mask))
    return loss, metrics
