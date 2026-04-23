import torch

from llamafactory.train.todi import compute_todi_loss


def test_todi_loss_is_zero_when_student_matches_teacher():
    logits = torch.tensor(
        [
            [
                [1.2, -0.4, 0.1],
                [0.3, 0.2, -0.5],
                [0.7, -0.3, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)

    loss, metrics = compute_todi_loss(logits, logits, labels, beta=1.0)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert metrics["todi_loss"] == 0.0


def test_todi_loss_becomes_jeffreys_divergence_when_beta_is_zero():
    student_logits = torch.tensor(
        [
            [
                [1.5, 0.1, -0.3],
                [0.0, 1.0, -0.2],
                [0.2, -0.1, 0.8],
            ]
        ],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [
            [
                [0.4, 1.1, -0.2],
                [0.8, -0.5, 0.3],
                [0.1, 0.7, -0.4],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 2, 1]], dtype=torch.long)

    loss, metrics = compute_todi_loss(student_logits, teacher_logits, labels, beta=0.0)

    shift_student = student_logits[..., :-1, :]
    shift_teacher = teacher_logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_mask = shift_labels.ne(-100)
    student_log_probs = torch.log_softmax(shift_student, dim=-1)
    teacher_log_probs = torch.log_softmax(shift_teacher, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    fkl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    rkl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    expected = (0.5 * (fkl + rkl) * valid_mask).sum() / valid_mask.sum()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.tensor(metrics["todi_alpha_mean"]), torch.tensor(0.5))
