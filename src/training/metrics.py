"""FAIR-Universe Weak Lensing scoring.

Output convention (B, 4): pred[:, :2] = μ, pred[:, 2:] = log σ.
Labels are pre-standardised by the dataset.
"""

from __future__ import annotations

from typing import Optional

import torch


def _split(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = pred[:, :2]
    log_sigma = pred[:, 2:]
    return mu, log_sigma, torch.exp(log_sigma)


def score_loss_fn(
    pred: torch.Tensor, target: torch.Tensor, lam: float = 1e3
) -> torch.Tensor:
    mu, log_sigma, _ = _split(pred)
    sq_err = (mu - target) ** 2
    # σ is trained on the residual via the NLL, but μ is detached there so its
    # gradient comes only from the λ·MSE term — the two heads stop fighting and
    # σ calibrates cleanly. The reported leaderboard score is unchanged.
    sq_err_sigma = (mu.detach() - target) ** 2
    nll = sq_err_sigma * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
    return (nll + lam * sq_err).sum(dim=1).mean()


@torch.no_grad()
def eval_metric_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1e3,
    label_std: Optional[torch.Tensor] = None,
) -> dict:
    mu, log_sigma, sigma = _split(pred)
    sq_err = (mu - target) ** 2
    chi2 = sq_err * torch.exp(-2.0 * log_sigma)
    score_loss = (chi2 + 2.0 * log_sigma + lam * sq_err).sum(dim=1)
    inside = (target - mu).abs() <= sigma

    out = {
        "score_loss": score_loss.sum(),
        "mse": sq_err.sum() / sq_err.shape[1],
        "coverage": inside.float().mean(dim=1).sum(),
    }

    if label_std is not None:
        s = label_std.to(pred.device, dtype=pred.dtype)
        log_s2 = 2.0 * torch.log(s)
        score_loss_orig = (
            chi2 + 2.0 * log_sigma + log_s2 + lam * sq_err * s**2
        ).sum(dim=1)
        out["score_loss_original"] = score_loss_orig.sum()

    return out
