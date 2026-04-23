import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .akl import _align_shared_vocab, _sanitize_logits


class ABKL(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def _compute_divergence(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        inf_mask: torch.Tensor,
    ) -> torch.Tensor:
        if math.isclose(self.alpha, 0.0, abs_tol=self.eps) and math.isclose(self.beta, 0.0, abs_tol=self.eps):
            log_diff = torch.log(student_probs.clamp_min(self.eps)) - torch.log(teacher_probs.clamp_min(self.eps))
            log_diff = log_diff.masked_fill(inf_mask, 0.0)
            return 0.5 * torch.sum(log_diff.square(), dim=-1)

        if math.isclose(self.alpha, 0.0, abs_tol=self.eps):
            q_beta = student_probs.pow(self.beta).masked_fill(inf_mask, 0.0)
            p_beta = teacher_probs.pow(self.beta).masked_fill(inf_mask, 0.0)
            likelihood_ratio = q_beta / p_beta.clamp_min(self.eps)
            term = q_beta * torch.log(likelihood_ratio.clamp_min(self.eps)) - q_beta + p_beta
            return (1.0 / self.beta) * torch.sum(term, dim=-1)

        if math.isclose(self.beta, 0.0, abs_tol=self.eps):
            p_alpha = teacher_probs.pow(self.alpha).masked_fill(inf_mask, 0.0)
            q_alpha = student_probs.pow(self.alpha).masked_fill(inf_mask, 0.0)
            term = p_alpha * torch.log((p_alpha / q_alpha.clamp_min(self.eps)).clamp_min(self.eps)) - p_alpha + q_alpha
            return (1.0 / self.alpha) * torch.sum(term, dim=-1)

        if math.isclose(self.alpha + self.beta, 0.0, abs_tol=self.eps):
            p_alpha = teacher_probs.pow(self.alpha).masked_fill(inf_mask, 0.0)
            q_alpha = student_probs.pow(self.alpha).masked_fill(inf_mask, 0.0)
            ratio = q_alpha / p_alpha.clamp_min(self.eps)
            term = (1.0 / self.alpha) * (torch.log(ratio.clamp_min(self.eps)) + ratio.clamp_min(self.eps).reciprocal() - 1.0)
            return torch.sum(term, dim=-1)

        p_alpha = teacher_probs.pow(self.alpha)
        q_beta = student_probs.pow(self.beta)
        p_alpha_beta = teacher_probs.pow(self.alpha + self.beta)
        q_alpha_beta = student_probs.pow(self.alpha + self.beta)

        first_term = (p_alpha * q_beta).masked_fill(inf_mask, 0.0)
        second_term = ((self.alpha / (self.alpha + self.beta)) * p_alpha_beta).masked_fill(inf_mask, 0.0)
        third_term = ((self.beta / (self.alpha + self.beta)) * q_alpha_beta).masked_fill(inf_mask, 0.0)
        return -torch.sum(first_term - second_term - third_term, dim=-1) / (self.alpha * self.beta)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_logits, student_inf_mask = _sanitize_logits(student_logits)
        teacher_logits, teacher_inf_mask = _sanitize_logits(teacher_logits)
        inf_mask = student_inf_mask | teacher_inf_mask

        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_probs_no_temp = F.softmax(student_logits, dim=-1, dtype=torch.float32)

        divergence = self._compute_divergence(student_probs, teacher_probs, inf_mask)
        entropy_temp = -torch.sum(student_probs * torch.log(student_probs.clamp_min(self.eps)), dim=-1)
        entropy_no_temp = -torch.sum(student_probs_no_temp * torch.log(student_probs_no_temp.clamp_min(self.eps)), dim=-1)
        return divergence.mean(), entropy_temp.mean(), entropy_no_temp.mean()


def compute_abkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    student_logits, teacher_logits, labels = _align_shared_vocab(student_logits, teacher_logits, labels, ignore_index)

    valid_mask = labels.ne(ignore_index)
    if not valid_mask.any():
        zero = student_logits.new_zeros(())
        return zero, {
            "abkl_loss": 0.0,
            "abkl_entropy_temp": 0.0,
            "abkl_entropy_no_temp": 0.0,
        }

    abkl = ABKL(temperature=temperature, alpha=alpha, beta=beta)
    loss, entropy_temp, entropy_no_temp = abkl(student_logits[valid_mask], teacher_logits[valid_mask])
    return loss, {
        "abkl_loss": loss.detach().item(),
        "abkl_entropy_temp": entropy_temp.detach().item(),
        "abkl_entropy_no_temp": entropy_no_temp.detach().item(),
    }
