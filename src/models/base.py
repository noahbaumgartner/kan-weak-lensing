from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping


class BaseKANModel(ABC):
    model: torch.nn.Module | None = None
    device: str = "cpu"

    @abstractmethod
    def build(self, device: str = "cpu") -> None:
        """Construct the underlying model from config."""

    def regularization_loss(self) -> float:
        return 0.0

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self) -> torch.nn.Module:
        return self.model

    def fit(
        self,
        dataset,
        epochs,
        optimizer_factory,
        loss_fn,
        batch_size,
        lamb,
        task_type="regression",
        epoch_callback=None,
        extra_eval_metrics_fn=None,
        grad_clip=None,
        early_stopping=False,
        es_patience=10,
        es_min_delta=0.0,
        es_restore_best=True,
        **kwargs,
    ):
        # Datasets return CPU tensors; the DataLoader moves each batch to
        # the model's device on demand. Preloading the full dataset to GPU
        # OOMs for larger sets like MNIST.
        #
        # Two input conventions are supported:
        #   1. legacy: dataset["train_input"] / ["train_label"] are tensors
        #      and we wrap them in a TensorDataset here.
        #   2. lazy:   dataset["train_input"] is already a torch Dataset
        #      yielding (x, y) — used by datasets that can't fit fully in
        #      RAM (e.g. weak_lensing). Labels live on the Dataset itself.
        # Validation uses "val_input" if provided, else falls back to
        # "test_input" (the older convention).
        ti = dataset["train_input"]
        vi = dataset.get("val_input", dataset.get("test_input"))
        if isinstance(ti, Dataset):
            train_ds = ti
        else:
            train_ds = TensorDataset(ti, dataset["train_label"])
        if isinstance(vi, Dataset):
            val_ds = vi
        else:
            vl = dataset.get("val_label", dataset.get("test_label"))
            val_ds = TensorDataset(vi, vl)

        n_train = len(train_ds)
        bs = n_train if (batch_size == -1 or batch_size >= n_train) else batch_size

        pin = self.device != "cpu"
        nw = int(kwargs.get("num_workers", 0))
        loader_kwargs = dict(pin_memory=pin, num_workers=nw)
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(kwargs.get("prefetch_factor", 4))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)

        opt = optimizer_factory(self.get_model().parameters())
        is_lbfgs = isinstance(opt, torch.optim.LBFGS)
        # Gradient clipping (max L2 norm). Off by default; datasets with a
        # large-scale loss (e.g. weak_lensing's λ=1e3 score_inference) set
        # ``training.grad_clip`` to keep updates from diverging to NaN.
        clip = float(grad_clip) if grad_clip else None

        results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            results["train_acc"] = []
            results["test_acc"] = []
        # Extra eval metrics get one list per key, populated on the val
        # set after each epoch.  Keys are discovered from the first batch.
        extra_metric_keys: list[str] = []

        device = self.device
        stopper = EarlyStopping(
            patience=es_patience,
            min_delta=es_min_delta,
            restore_best=es_restore_best,
            enabled=early_stopping,
        )
        for epoch in tqdm(range(epochs), desc="Training"):
            # --- Train (accumulate metrics during the step to skip a redundant eval pass) ---
            self.get_model().train()
            train_loss_sum = torch.zeros((), device=device)
            train_correct_sum = torch.zeros((), device=device)
            train_total = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if is_lbfgs:
                    captured = {}

                    def closure():
                        opt.zero_grad(set_to_none=True)
                        pred = self.predict(x)
                        data_loss = loss_fn(pred, y)
                        captured["pred"] = pred.detach()
                        captured["data_loss"] = data_loss.detach()
                        reg = self.regularization_loss()
                        loss = data_loss + lamb * reg if reg > 0 else data_loss
                        loss.backward()
                        if clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.get_model().parameters(), clip
                            )
                        return loss

                    opt.step(closure)
                    pred_detached = captured["pred"]
                    data_loss = captured["data_loss"]
                else:
                    opt.zero_grad(set_to_none=True)
                    pred = self.predict(x)
                    data_loss = loss_fn(pred, y)
                    reg = self.regularization_loss()
                    loss = data_loss + lamb * reg if reg > 0 else data_loss
                    loss.backward()
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.get_model().parameters(), clip
                        )
                    opt.step()
                    pred_detached = pred.detach()
                    data_loss = data_loss.detach()

                bs_x = x.shape[0]
                train_loss_sum = train_loss_sum + data_loss * bs_x
                if task_type == "classification":
                    train_correct_sum = train_correct_sum + (pred_detached.argmax(dim=1) == y).sum()
                train_total += bs_x

            train_mse = (train_loss_sum / train_total).item()

            # --- Validate (on val set only) ---
            self.get_model().eval()
            val_loss_sum = torch.zeros((), device=device)
            val_correct_sum = torch.zeros((), device=device)
            val_total = 0
            extra_sums: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = self.predict(x)
                    bs_x = x.shape[0]
                    val_loss_sum = val_loss_sum + loss_fn(pred, y) * bs_x
                    if task_type == "classification":
                        val_correct_sum = val_correct_sum + (pred.argmax(dim=1) == y).sum()
                    if extra_eval_metrics_fn is not None:
                        for k, v in extra_eval_metrics_fn(pred, y).items():
                            extra_sums[k] = extra_sums.get(k, torch.zeros((), device=device)) + v
                    val_total += bs_x
            val_mse = (val_loss_sum / val_total).item()

            if extra_eval_metrics_fn is not None:
                for k, total in extra_sums.items():
                    if k not in extra_metric_keys:
                        extra_metric_keys.append(k)
                        results[k] = []
                    results[k].append((total / val_total).item())

            results["train_loss"].append(train_mse)
            results["test_loss"].append(val_mse)
            results["reg"].append(0.0)

            if task_type == "classification":
                results["train_acc"].append((train_correct_sum / train_total).item())
                results["test_acc"].append((val_correct_sum / val_total).item())

            if epoch_callback is not None:
                epoch_callback(epoch, self)

            if stopper.step(epoch, val_mse, self.get_model()):
                break

        stopper.finalize(self.get_model(), results, device)
        return results
