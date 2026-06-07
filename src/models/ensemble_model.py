from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .base import BaseKANModel


class EnsembleModel(BaseKANModel):
    """Deep ensemble of KAN regressors → Gaussian posterior.

    Versuch "ensemble": train ``n_members`` independent member models with the
    *same* architecture but *different* random initialisation, each on plain
    MSE against the cosmological parameters (Om, S8). At eval / predict time the
    member point-predictions are combined into a Gaussian:

        mu        = mean over members      (per parameter)
        log_sigma = log(std over members)  (per parameter)

    i.e. the spread between the differently-initialised members *is* the
    predicted uncertainty (Gaussian assumption). ``predict`` returns the
    4-vector ``[mu_Om, mu_S8, log_sigma_Om, log_sigma_S8]`` expected by the
    FAIR-Universe score / 68%-coverage metrics, so this plugs straight into the
    score-style logging path of the Trainer.

    The member is any ordinary single-model wrapper (default: FastKANModel),
    supplied already-constructed via Hydra and cloned ``n_members`` times.
    """

    def __init__(
        self,
        member: BaseKANModel,
        n_members: int = 8,
        seed_base: int = 0,
        sigma_floor: float = 1e-3,
        **kwargs,
    ):
        # ``member`` is an un-built BaseKANModel wrapper (model is still None);
        # we deep-copy it per ensemble member in build().
        self._member_proto = member
        self.n_members = int(n_members)
        self.seed_base = int(seed_base)
        self.sigma_floor = float(sigma_floor)
        self.members: list[BaseKANModel] = []
        self.model: nn.Module | None = None

    # ------------------------------------------------------------------ build
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
            member.build(device=device)
            self.members.append(member)
            mods.append(member.get_model())
        # A ModuleList gives a single handle for parameters()/to()/train()/eval().
        self.model = nn.ModuleList(mods).to(device)

    # --------------------------------------------------------------- combine
    def _member_preds(self, x: torch.Tensor) -> torch.Tensor:
        # (n_members, B, P) with P = number of predicted parameters (2)
        return torch.stack([m.predict(x) for m in self.members], dim=0)

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        preds = self._member_preds(x)            # (N, B, P)
        mu = preds.mean(dim=0)                    # (B, P)
        if preds.shape[0] > 1:
            std = preds.std(dim=0, unbiased=True)
        else:
            std = torch.full_like(mu, self.sigma_floor)
        std = std.clamp_min(self.sigma_floor)
        log_sigma = torch.log(std)
        return torch.cat([mu, log_sigma], dim=1)  # (B, 2P)

    # -------------------------------------------------------------------- fit
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
        **kwargs,
    ):
        """Train all members on MSE; log score-style metrics on the val set.

        Each member is trained on its own MSE; the per-batch objective is the
        sum of member MSEs, so members stay independent (their parameter sets
        don't interact) and diversity comes from the different inits. The val
        metrics are computed on the *combined* Gaussian prediction, reusing the
        Trainer-supplied ``loss_fn`` (score_inference) and
        ``extra_eval_metrics_fn`` (eval_metric_sums) for full consistency with
        the single-model score Versuch.
        """
        device = self.device

        # --- data handling (mirrors BaseKANModel.fit) ---
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
        pin = device != "cpu"
        nw = int(kwargs.get("num_workers", 0))
        loader_kwargs = dict(pin_memory=pin, num_workers=nw)
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(kwargs.get("prefetch_factor", 4))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)

        opt = optimizer_factory(self.model.parameters())
        if isinstance(opt, torch.optim.LBFGS):
            raise ValueError("EnsembleModel does not support the LBFGS optimizer; use adam/adamw/sgd.")
        clip = float(grad_clip) if grad_clip else None
        mse = lambda p, t: torch.mean((p - t) ** 2)

        results: dict[str, list] = {
            "train_loss": [], "test_loss": [], "reg": [],
            "mse": [], "coverage": [], "score_loss": [],
        }
        extra_metric_keys: list[str] = []

        for epoch in tqdm(range(epochs), desc="Training (ensemble)"):
            # --- train: each member on its own MSE ---
            self.model.train()
            train_loss_sum = torch.zeros((), device=device)
            train_total = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                batch_loss = torch.zeros((), device=device)
                for m in self.members:
                    batch_loss = batch_loss + mse(m.predict(x), y)
                batch_loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                opt.step()
                bs_x = x.shape[0]
                # report the mean per-member MSE (comparable to single-model runs)
                train_loss_sum = train_loss_sum + (batch_loss.detach() / self.n_members) * bs_x
                train_total += bs_x
            train_mse = (train_loss_sum / train_total).item()

            # --- validate on the combined Gaussian prediction ---
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
            results["reg"].append(0.0)
            if extra_eval_metrics_fn is not None:
                for k, total in extra_sums.items():
                    if k not in extra_metric_keys:
                        extra_metric_keys.append(k)
                        results.setdefault(k, [])
                    results[k].append((total / val_total).item())

            if epoch_callback is not None:
                epoch_callback(epoch, self)

        return results
