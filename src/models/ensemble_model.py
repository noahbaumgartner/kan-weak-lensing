import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .base import BaseKANModel
from src.training.early_stopping import EarlyStopping


class EnsembleModel(BaseKANModel):
    """Bagging ensemble of KAN regressors -> Gaussian posterior via member spread.

    Each member trains on its own data subset (subset_fraction/bootstrap) and
    optionally a jittered grid resolution (grid_jitter), so the eval-time
    spread across members is a statistically grounded uncertainty, not just
    init noise. predict() returns [mu_Om, mu_S8, log_sigma_Om, log_sigma_S8].
    """

    # grid-resolution attr per member wrapper (num_grids/grid_size; WavKAN has neither)
    _GRID_ATTRS = ("num_grids", "grid_size")

    def __init__(
        self,
        member: BaseKANModel,
        n_members: int = 8,
        seed_base: int = 0,
        sigma_floor: float = 1e-3,
        subset_fraction: float = 0.8,
        bootstrap: bool = False,
        subset_seed: int = 0,
        grid_jitter: int = 0,
        **kwargs,
    ):
        # unbuilt wrapper, deep-copied per member in build()
        self._member_proto = member
        self.n_members = int(n_members)
        self.seed_base = int(seed_base)
        self.sigma_floor = float(sigma_floor)
        self.subset_fraction = float(subset_fraction)
        self.bootstrap = bool(bootstrap)
        self.subset_seed = int(subset_seed)
        self.grid_jitter = int(grid_jitter)
        self.members: list[BaseKANModel] = []
        self.model: nn.Module | None = None

    def _apply_grid_jitter(self, member: BaseKANModel, seed: int) -> None:
        """Perturb member's grid resolution by a reproducible +/- grid_jitter offset."""
        rng = random.Random(seed)
        offset = rng.randint(-self.grid_jitter, self.grid_jitter)
        for attr in self._GRID_ATTRS:
            if hasattr(member, attr):
                setattr(member, attr, max(1, getattr(member, attr) + offset))
                break

    def build(self, device: str = "cpu") -> None:
        self.device = device
        self.members = []
        mods = []
        for i in range(self.n_members):
            seed = self.seed_base + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            member = copy.deepcopy(self._member_proto)
            if self.grid_jitter:
                self._apply_grid_jitter(member, seed)
            member.build(device=device)
            self.members.append(member)
            mods.append(member.get_model())
        # A ModuleList gives a single handle for parameters()/to()/train()/eval().
        self.model = nn.ModuleList(mods).to(device)

    def _member_preds(self, x: torch.Tensor) -> torch.Tensor:
        # (n_members, B, P) with P = number of predicted parameters (2)
        return torch.stack([m.predict(x) for m in self.members], dim=0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        preds = self._member_preds(x)            # (N, B, P)
        mu = preds.mean(dim=0)                    # (B, P)
        if preds.shape[0] > 1:
            std = preds.std(dim=0, unbiased=True)
        else:
            std = torch.full_like(mu, self.sigma_floor)
        std = std.clamp_min(self.sigma_floor)
        log_sigma = torch.log(std)
        return torch.cat([mu, log_sigma], dim=1)  # (B, 2P)

    def fit(
        self,
        dataset,
        epochs,
        optimizer_factory,
        loss_fn,
        batch_size,
        lamb,
        extra_eval_metrics_fn=None,
        **kwargs,
    ):
        """Train each member on its own data subset; validate on the combined Gaussian prediction."""
        device = self.device

        # mirrors BaseKANModel.fit
        train_ds = dataset["train_input"]
        val_ds = dataset["val_input"]

        n_train = len(train_ds)
        bs = min(batch_size, n_train)
        pin = device != "cpu"
        nw = int(kwargs.get("num_workers", 0))
        loader_kwargs = dict(pin_memory=pin, num_workers=nw)
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(kwargs.get("prefetch_factor", 4))

        # per-member resample (bagging) so members are ~independent estimators
        member_loaders = []
        for i in range(self.n_members):
            gen = torch.Generator().manual_seed(self.subset_seed + i)
            if self.bootstrap:
                # full-size sample with replacement (~63% unique points)
                member_idx = torch.randint(0, n_train, (n_train,), generator=gen)
            else:
                # random fraction without replacement
                k = max(1, int(round(n_train * self.subset_fraction)))
                member_idx = torch.randperm(n_train, generator=gen)[:k]
            member_ds = Subset(train_ds, member_idx.tolist())
            n_sub = len(member_ds)
            sub_bs = min(batch_size, n_sub)
            member_loaders.append(
                DataLoader(member_ds, batch_size=sub_bs, shuffle=True, **loader_kwargs)
            )
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)

        opt = optimizer_factory(self.model.parameters())
        mse = lambda p, t: torch.mean((p - t) ** 2)

        results: dict[str, list] = {
            "train_loss": [], "test_loss": [],
            "mse": [], "coverage": [], "score_loss": [],
        }
        extra_metric_keys: list[str] = []

        stopper = EarlyStopping(
            patience=kwargs.get("es_patience", 10),
            min_delta=kwargs.get("es_min_delta", 0.0),
            restore_best=kwargs.get("es_restore_best", True),
            enabled=kwargs.get("early_stopping", False),
        )

        for epoch in tqdm(range(epochs), desc="Training (ensemble)"):
            self.model.train()
            member_mse_sum = torch.zeros((), device=device)
            for m, m_loader in zip(self.members, member_loaders):
                loss_sum = torch.zeros((), device=device)
                seen = 0
                for x, y in m_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    # only this member's params receive gradients this step
                    batch_loss = mse(m.predict(x), y)
                    batch_loss.backward()
                    opt.step()
                    bs_x = x.shape[0]
                    loss_sum = loss_sum + batch_loss.detach() * bs_x
                    seen += bs_x
                member_mse_sum = member_mse_sum + loss_sum / max(seen, 1)
            # report the mean per-member training MSE (comparable to single-model runs)
            train_mse = (member_mse_sum / self.n_members).item()

            # validate on the combined Gaussian prediction
            self.model.eval()
            val_loss_sum = torch.zeros((), device=device)
            val_total = 0
            extra_sums: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = self.predict(x)                      # (B, 4)
                    bs_x = x.shape[0]
                    val_loss_sum = val_loss_sum + loss_fn(pred, y) * bs_x
                    if extra_eval_metrics_fn is not None:
                        for k, v in extra_eval_metrics_fn(pred, y).items():
                            extra_sums[k] = extra_sums.get(k, torch.zeros((), device=device)) + v
                    val_total += bs_x
            val_loss = (val_loss_sum / val_total).item()

            results["train_loss"].append(train_mse)
            results["test_loss"].append(val_loss)
            if extra_eval_metrics_fn is not None:
                for k, total in extra_sums.items():
                    if k not in extra_metric_keys:
                        extra_metric_keys.append(k)
                        results.setdefault(k, [])
                    results[k].append((total / val_total).item())

            if stopper.step(epoch, val_loss, self.model):
                break

        stopper.finalize(self.model, results, device)
        return results
