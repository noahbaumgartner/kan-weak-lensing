"""Validation-loss early stopping shared by the single-model and ensemble
training loops.

Monitors the per-epoch validation loss (the same quantity the Optuna sweeper
minimises). When it stops improving for ``patience`` epochs, training halts; the
best weights are restored and the per-epoch result lists are truncated to the
best epoch. That truncation keeps every downstream consumer consistent with the
model actually kept: the Trainer's ``final_*`` metrics (read from ``[-1]``) and
main.py's returned objective (``test_loss[-1]``) then both reflect the best
epoch rather than the post-plateau tail.

``min_delta`` is a *relative* threshold: an epoch counts as an improvement only
if it beats the running best by at least ``min_delta * abs(best)`` (e.g.
``0.005`` -> 0.5%). Relative (not absolute) because the monitored loss lives on
wildly different scales per objective — raw-label MSE is O(0.1-1) while the
score_inference loss carries a λ=1e3 penalty — so no single absolute delta would
suit both. ``min_delta == 0`` reduces to "any decrease counts". The restored /
truncated epoch is the last min_delta-qualified best, so the kept model, the
logged finals and the returned objective all agree.
"""

from __future__ import annotations

import math


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, restore_best=True, enabled=True):
        self.patience = int(patience) if patience is not None else None
        self.min_delta = float(min_delta)
        self.restore_best = bool(restore_best)
        self.enabled = bool(enabled)
        self.best = float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.num_bad = 0

    def step(self, epoch: int, value: float, model) -> bool:
        """Record this epoch's monitored value. Returns True if training should
        stop after this epoch."""
        if not self.enabled:
            return False
        # Relative improvement: beat the running best by at least min_delta of
        # its magnitude. The first finite value always wins (best starts at inf).
        threshold = self.best - self.min_delta * abs(self.best)
        if not math.isfinite(self.best) or value < threshold:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
            if self.restore_best:
                # Park a CPU copy so the snapshot doesn't pin GPU memory.
                self.best_state = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in model.state_dict().items()
                }
        else:
            self.num_bad += 1
        return self.patience is not None and self.num_bad >= self.patience

    def finalize(self, model, results: dict, device) -> None:
        """Restore the best weights and truncate every result list to the best
        epoch. No-op when disabled or when no epoch ever improved."""
        if not self.enabled or self.best_epoch < 0:
            return
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in self.best_state.items()}
            )
        cut = self.best_epoch + 1
        for v in results.values():
            if isinstance(v, list):
                del v[cut:]
