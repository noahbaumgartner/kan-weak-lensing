from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm

from .base import BaseKANModel


class EnsembleModel(BaseKANModel):
    """Bagging ensemble of KAN regressors → Gaussian posterior.

    Versuch "ensemble": train ``n_members`` independent member models with the
    *same* architecture, each on plain MSE against the cosmological parameters
    (Om, S8). Crucially, each member is trained on its *own random subset* of
    the training data (resampling / bagging), so the members are approximately
    independent draws from the sampling distribution of the estimator. At eval
    / predict time the member point-predictions are combined into a Gaussian:

        mu        = mean over members      (per parameter)
        log_sigma = log(std over members)  (per parameter)

    Because each member sees different data, the spread between members
    estimates how much the prediction would change under a different training
    set — i.e. the std *is* a statistically grounded predictive uncertainty
    (Gaussian assumption), not merely initialisation noise. ``predict`` returns
    the 4-vector ``[mu_Om, mu_S8, log_sigma_Om, log_sigma_S8]`` expected by the
    FAIR-Universe score / 68%-coverage metrics, so this plugs straight into the
    score-style logging path of the Trainer.

    Data resampling per member is controlled by:
        ``subset_fraction`` — fraction of the training set each member sees
                              (subsampling without replacement; ignored when
                              ``bootstrap=True``).
        ``bootstrap``       — if True, draw a full-size sample *with*
                              replacement instead (classic bootstrap bagging;
                              each member sees ~63% unique points).
    Member i draws its subset with seed ``subset_seed + i`` (and is also
    initialised with ``seed_base + i``), so diversity comes from both the data
    and the init while staying fully reproducible.

    The member is any ordinary single-model wrapper (default: FastKANModel),
    supplied already-constructed via Hydra and cloned ``n_members`` times.
    """

    def __init__(
        self,
        member: BaseKANModel,
        n_members: int = 8,
        seed_base: int = 0,
        sigma_floor: float = 1e-3,
        subset_fraction: float = 0.8,
        bootstrap: bool = False,
        subset_seed: int = 0,
        **kwargs,
    ):
        # ``member`` is an un-built BaseKANModel wrapper (model is still None);
        # we deep-copy it per ensemble member in build().
        self._member_proto = member
        self.n_members = int(n_members)
        self.seed_base = int(seed_base)
        self.sigma_floor = float(sigma_floor)
        self.subset_fraction = float(subset_fraction)
        self.bootstrap = bool(bootstrap)
        self.subset_seed = int(subset_seed)
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
        **kwargs,
    ):
        """Train each member on its own data subset; log score-style metrics.

        Every member is trained independently on plain MSE, but over a
        *different random subset* of the training set (subsampling, or
        bootstrap resampling when ``bootstrap=True``). Members therefore differ
        in both their data and their init, which is what lets the eval-time
        spread between members be read as a real predictive standard deviation
        (bagging) rather than mere initialisation noise. The val metrics are
        computed on the *combined* Gaussian prediction, reusing the
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

        # --- per-member data subsets (bagging) ---
        # Each member sees its own resample of the training set so the members
        # are approximately independent estimators; their eval-time spread is
        # then a statistically grounded uncertainty.
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
            sub_bs = n_sub if (batch_size == -1 or batch_size >= n_sub) else batch_size
            member_loaders.append(
                DataLoader(member_ds, batch_size=sub_bs, shuffle=True, **loader_kwargs)
            )
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)

        opt = optimizer_factory(self.model.parameters())
        if isinstance(opt, torch.optim.LBFGS):
            raise ValueError("EnsembleModel does not support the LBFGS optimizer; use adam/adamw/sgd.")
        mse = lambda p, t: torch.mean((p - t) ** 2)

        results: dict[str, list] = {
            "train_loss": [], "test_loss": [], "reg": [],
            "mse": [], "coverage": [], "score_loss": [],
        }
        extra_metric_keys: list[str] = []

        for epoch in tqdm(range(epochs), desc="Training (ensemble)"):
            # --- train: each member on its own data subset / MSE ---
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
