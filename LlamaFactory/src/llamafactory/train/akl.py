from typing import Optional

import torch
import torch.nn.functional as F


def _sanitize_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inf_mask = torch.isinf(logits)
    if inf_mask.any():
        logits = logits.masked_fill(inf_mask, 0.0)

    return logits.to(torch.float32), inf_mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_values = values.masked_select(mask)
    if masked_values.numel() == 0:
        return values.new_zeros(())

    return masked_values.mean()


def _align_shared_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_logits.size(-1) == teacher_logits.size(-1):
        return student_logits, teacher_logits, labels

    # Some checkpoints append extra special tokens at the end of the vocab.
    # AKL only compares the shared prefix vocab, and skips labels outside it.
    shared_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[..., :shared_vocab_size]
    teacher_logits = teacher_logits[..., :shared_vocab_size]
    labels = labels.masked_fill(labels >= shared_vocab_size, ignore_index)
    return student_logits, teacher_logits, labels


def get_adaptive_ratios(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mu: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32).detach()

        sorted_teacher_probs, sorted_indices = teacher_probs.sort(dim=-1, descending=True)
        sorted_student_probs = student_probs.gather(dim=-1, index=sorted_indices)

        errors = torch.abs(sorted_teacher_probs - sorted_student_probs)
        cumulative_teacher_probs = torch.cumsum(sorted_teacher_probs, dim=-1)
        tail_mask = cumulative_teacher_probs > mu
        tail_mask[..., 0] = False

        head_error = errors.masked_fill(tail_mask, 0.0).sum(dim=-1)
        tail_error = errors.masked_fill(~tail_mask, 0.0).sum(dim=-1)
        total_error = head_error + tail_error
        fallback_ratio = torch.full_like(head_error, 0.5)

        high_ratio = torch.where(total_error > 0, head_error / total_error, fallback_ratio)
        low_ratio = torch.where(total_error > 0, tail_error / total_error, fallback_ratio)
        return high_ratio, low_ratio


def _weighted_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    inf_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    ratio: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)

    teacher_entropy = (teacher_probs * teacher_log_probs).masked_fill(inf_mask, 0.0).sum(dim=-1)
    cross_entropy = (teacher_probs * student_log_probs).masked_fill(inf_mask, 0.0).sum(dim=-1)
    token_kl = teacher_entropy - cross_entropy

    valid_mask = valid_mask.to(token_kl.dtype)
    if ratio is not None:
        token_kl = token_kl * ratio.to(token_kl.dtype)

    normalizer = valid_mask.sum() + 1e-8
    return (token_kl * valid_mask).sum() / normalizer, token_kl


def compute_akl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    mu: float = 0.5,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    student_logits, teacher_logits, labels = _align_shared_vocab(student_logits, teacher_logits, labels, ignore_index)

    student_logits, student_inf_mask = _sanitize_logits(student_logits)
    teacher_logits, teacher_inf_mask = _sanitize_logits(teacher_logits)
    inf_mask = student_inf_mask | teacher_inf_mask
    valid_mask = labels.ne(ignore_index)

    high_ratio, low_ratio = get_adaptive_ratios(teacher_logits, student_logits, mu=mu)
    forward_loss, _ = _weighted_kl_divergence(teacher_logits, student_logits, inf_mask, valid_mask, high_ratio)
    reverse_loss, _ = _weighted_kl_divergence(student_logits, teacher_logits, inf_mask, valid_mask, low_ratio)
    loss = forward_loss + reverse_loss

    metrics = {
        "akl_loss": loss.detach().item(),
        "akl_forward_loss": forward_loss.detach().item(),
        "akl_reverse_loss": reverse_loss.detach().item(),
        "akl_high_ratio": _masked_mean(high_ratio, valid_mask).item(),
        "akl_low_ratio": _masked_mean(low_ratio, valid_mask).item(),
    }
    return loss, metrics
