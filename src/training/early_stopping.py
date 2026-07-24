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
        """Record this epoch's value. Returns True if training should stop after this epoch."""
        if not self.enabled:
            return False
        threshold = self.best - self.min_delta * abs(self.best)
        if not math.isfinite(self.best) or value < threshold:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
            if self.restore_best:
                # CPU copy so the snapshot doesn't pin GPU memory
                self.best_state = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in model.state_dict().items()
                }
        else:
            self.num_bad += 1
        return self.patience is not None and self.num_bad >= self.patience

    def finalize(self, model, results: dict, device) -> None:
        """Restore the best weights and truncate every result list to the best epoch."""
        if not self.enabled or self.best_epoch < 0:
            return
        if self.restore_best and self.best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})
        cut = self.best_epoch + 1
        for v in results.values():
            if isinstance(v, list):
                del v[cut:]
