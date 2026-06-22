import random
import time

import numpy as np
import torch
import mlflow
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import get_device
from src.training.metrics import (
    eval_metric_sums,
    regression_metric_sums,
    score_inference_loss,
)

# Friendly per-parameter names for the 2 cosmological targets (Ω_m, S_8).
_PARAM_NAMES = ["om", "s8"]


def _param_names(n: int) -> list[str]:
    return _PARAM_NAMES[:n] if n == len(_PARAM_NAMES) else [f"p{j}" for j in range(n)]


def _label_var_per(obj, n_targets: int):
    """Per-parameter (column-wise) label variance, in the label's own space.

    Used for *honest* per-parameter R² = 1 − MSE_j / Var_j, which is invariant
    to label scaling — unlike the pooled variance, which mixes the Ω_m and S_8
    columns and inflates R²."""
    if obj is None:
        return None
    if not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    obj = obj.reshape(-1, obj.shape[-1])[:, :n_targets]
    return torch.var(obj, dim=0, unbiased=False).tolist()


def _per_param_metrics(mse_per: list[float], var_per) -> dict:
    """Build per-parameter R² / standardised-MSE metrics from per-parameter MSE.

    ``std_mse_j = MSE_j / Var_j = 1 − R²_j`` is the per-parameter analogue of the
    competition's "MEAN MSE (STANDARDIZED)"; the macro mean averages the two
    parameters with equal weight (no mean-gap inflation)."""
    names = _param_names(len(mse_per))
    out: dict[str, float] = {}
    r2s, std_mses = [], []
    for j, mse_j in enumerate(mse_per):
        out[f"mse_{names[j]}"] = mse_j
        if var_per is not None and var_per[j] > 0:
            r2_j = 1.0 - mse_j / var_per[j]
            std_mse_j = mse_j / var_per[j]
            out[f"r2_{names[j]}"] = r2_j
            out[f"mse_std_{names[j]}"] = std_mse_j
            r2s.append(r2_j)
            std_mses.append(std_mse_j)
    if r2s:
        out["r2_macro"] = sum(r2s) / len(r2s)
        out["mse_std_mean"] = sum(std_mses) / len(std_mses)
    return out


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        device_cfg = cfg.training.get("device", "auto")
        self.device = get_device() if device_cfg == "auto" else torch.device(device_cfg)
        print(f"Using device: {self.device}")
        if "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True

    def _create_loss_fn(self, task_type, loss_name: str | None = None):
        if task_type == "classification":
            return torch.nn.CrossEntropyLoss()
        if loss_name == "score_inference":
            # FAIR-Universe weak-lensing: minimise the negative of the
            # PDF leaderboard score with λ=1e3.  Labels must be
            # standardised by the dataset so σ lives at a natural O(1)
            # scale.  No gradient clipping / log σ clamping anymore — use a
            # small LR if the λ·MSE term destabilises training.
            return score_inference_loss
        return lambda pred, target: torch.mean((pred - target) ** 2)

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        cfg = self.cfg
        task_type = cfg.dataset.get("task_type", "regression")

        # global seed
        self._seed_everything(cfg.get("seed", 42))

        # load dataset
        dataset_obj = instantiate(cfg.dataset)
        dataset = dataset_obj.create()

        # build model
        model = instantiate(cfg.model)
        model.build(device=self.device)

        # optimizer factory: call with model.parameters() to build the torch optimizer
        optimizer_factory = instantiate(cfg.optimizer)

        # create loss function — datasets can opt into a custom loss by
        # setting ``loss:`` in their YAML (e.g. weak_lensing -> score_inference).
        loss_name = cfg.dataset.get("loss", None)
        loss_fn = self._create_loss_fn(task_type, loss_name=loss_name)
        is_score_inference = loss_name == "score_inference"

        # setup MLflow
        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment", "experiment"))
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(1)

        with mlflow.start_run(run_name=_generate_run_name(cfg)):
            # log config parameters
            flat_cfg = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(flat_cfg)
            mlflow.log_param("parameter_count", model.parameter_count())
            mlflow.log_param("dataset", _get_dataset_name(cfg))
            mlflow.log_param("model", _get_model_name(cfg))
            mlflow.log_param("shape", _get_shape(cfg))

            # train
            t_start = time.time()

            fit_kwargs = dict(
                dataset=dataset,
                epochs=cfg.training.epochs,
                optimizer_factory=optimizer_factory,
                loss_fn=loss_fn,
                batch_size=cfg.training.get("batch_size", -1),
                lamb=cfg.training.get("lamb", 0.0),
                task_type=task_type,
                num_workers=cfg.training.get("num_workers", 0),
                prefetch_factor=cfg.training.get("prefetch_factor", 4),
                grad_clip=cfg.training.get("grad_clip", None),
                early_stopping=cfg.training.get("early_stopping", False),
                es_patience=cfg.training.get("es_patience", 10),
                es_min_delta=cfg.training.get("es_min_delta", 0.0),
                es_restore_best=cfg.training.get("es_restore_best", True),
            )

            if is_score_inference:
                label_stats = dataset.get("label_stats")
                if label_stats is not None:
                    label_std_t = torch.as_tensor(label_stats[1], dtype=torch.float32)
                    fit_kwargs["extra_eval_metrics_fn"] = (
                        lambda p, t: eval_metric_sums(p, t, label_std=label_std_t)
                    )
                else:
                    fit_kwargs["extra_eval_metrics_fn"] = eval_metric_sums
            elif task_type == "regression":
                # Plain-MSE objective: accumulate per-parameter squared error so
                # we can log honest per-parameter R² / standardised MSE below.
                fit_kwargs["extra_eval_metrics_fn"] = regression_metric_sums

            results = model.fit(**fit_kwargs)

            train_time = time.time() - t_start

            # R² needs the label variance. Datasets may expose labels
            # as tensors (legacy), numpy arrays (weak_lensing), or omit
            # them entirely (test-only set).
            def _label_var(obj):
                if obj is None:
                    return float("nan")
                if not isinstance(obj, torch.Tensor):
                    obj = torch.as_tensor(obj)
                return float(torch.var(obj, unbiased=False))

            # log metrics per step
            if task_type == "classification":
                for step_i, (tl, vl, ta, va) in enumerate(
                    zip(
                        results["train_loss"],
                        results["test_loss"],
                        results["train_acc"],
                        results["test_acc"],
                    )
                ):
                    mlflow.log_metrics(
                        {
                            "train_loss": float(tl),
                            "test_loss": float(vl),
                            "train_acc": float(ta),
                            "test_acc": float(va),
                        },
                        step=step_i,
                    )

                mlflow.log_metric("final_train_loss", float(results["train_loss"][-1]))
                mlflow.log_metric("final_test_loss", float(results["test_loss"][-1]))
                mlflow.log_metric("final_train_acc", float(results["train_acc"][-1]))
                mlflow.log_metric("final_test_acc", float(results["test_acc"][-1]))
            elif is_score_inference:
                # ``train_loss`` / ``test_loss`` are the PDF score-loss
                # (negative leaderboard score, λ=1e3) — i.e. the exact
                # quantity submitted to Codabench.  Labels are
                # standardised, so MSE/R²/score are reported in z-space;
                # multiply MSE component-wise by std² to recover original
                # units at submission time.
                n_steps = len(results["train_loss"])
                val_label = dataset.get("val_label", dataset.get("test_label"))
                test_var = _label_var(val_label)
                # Per-parameter variance (Ω_m, S_8) for honest per-parameter R².
                n_param = sum(1 for k in results if k.startswith("mse_p"))
                var_per = _label_var_per(val_label, n_param)
                # ``test_score`` is the Codabench leaderboard score — computed
                # in original Ω_m / S_8 units when labels were standardised,
                # otherwise identical to ``-score_loss``.
                score_key = "score_loss_original" if "score_loss_original" in results else "score_loss"
                for step_i in range(n_steps):
                    tl = float(results["train_loss"][step_i])
                    vl = float(results["test_loss"][step_i])
                    test_mse = float(results["mse"][step_i])
                    metrics = {
                        "train_loss": tl,            # z-space PDF score-loss (training objective)
                        "test_loss": vl,             # z-space PDF score-loss on val
                        "test_score": -float(results[score_key][step_i]),  # Codabench
                        "test_mse": test_mse,
                        "test_rmse": test_mse**0.5,
                        "test_coverage": float(results["coverage"][step_i]),
                    }
                    if test_var and test_var > 0:
                        metrics["test_r2"] = 1.0 - test_mse / test_var
                    mse_per = [float(results[f"mse_p{j}"][step_i]) for j in range(n_param)]
                    metrics.update(
                        {f"test_{k}": v for k, v in _per_param_metrics(mse_per, var_per).items()}
                    )
                    mlflow.log_metrics(metrics, step=step_i)

                # Final summary metrics (last epoch).
                final = {
                    k: v[-1]
                    for k, v in results.items()
                    if isinstance(v, list) and v
                }
                final_test_mse = float(final["mse"])
                final_test_score = -float(final[score_key])
                mlflow.log_metric("final_train_loss", float(final["train_loss"]))
                mlflow.log_metric("final_test_loss", float(final["test_loss"]))
                mlflow.log_metric("final_test_score", final_test_score)
                mlflow.log_metric("final_test_mse", final_test_mse)
                mlflow.log_metric("final_test_rmse", final_test_mse**0.5)
                mlflow.log_metric("final_test_coverage", float(final["coverage"]))
                if test_var and test_var > 0:
                    mlflow.log_metric(
                        "final_test_r2", 1.0 - final_test_mse / test_var
                    )

                final_mse_per = [float(final[f"mse_p{j}"]) for j in range(n_param)]
                pp = _per_param_metrics(final_mse_per, var_per)
                for k, v in pp.items():
                    mlflow.log_metric(f"final_test_{k}", v)

                print(f"\nFinal Test Score:        {final_test_score:.4f}  (Codabench scale)")
                print(f"Final Test MSE:          {final_test_mse:.6f}")
                print(f"Final Coverage:          {float(final['coverage']):.4f}  (target ~0.68)")
                if "r2_macro" in pp:
                    print(f"Per-param R² (honest):   {pp['r2_macro']:.4f}  (macro mean)")
                    print(f"Std-MSE (leaderboard):   {pp['mse_std_mean']:.4f}  (= 1 − R², lower=better)")
            else:
                # Per-parameter variance (Ω_m, S_8) for honest per-parameter R².
                # The pooled `final_test_r2` below mixes both columns and is
                # inflated when labels are raw (standardize_labels: false) — the
                # per-parameter R² / standardised MSE are the trustworthy numbers.
                n_param = sum(1 for k in results if k.startswith("mse_p"))
                var_per = _label_var_per(
                    dataset.get("val_label", dataset.get("test_label")), n_param
                )
                for step_i, (tl, vl) in enumerate(
                    zip(results["train_loss"], results["test_loss"])
                ):
                    step_metrics = {
                        "train_mse": float(tl),
                        "test_mse": float(vl),
                        "train_rmse": float(tl) ** 0.5,
                        "test_rmse": float(vl) ** 0.5,
                    }
                    if n_param:
                        mse_per = [float(results[f"mse_p{j}"][step_i]) for j in range(n_param)]
                        step_metrics.update(
                            {f"test_{k}": v for k, v in _per_param_metrics(mse_per, var_per).items()}
                        )
                    mlflow.log_metrics(step_metrics, step=step_i)

                final_train_mse = float(results["train_loss"][-1])
                final_test_mse = float(results["test_loss"][-1])
                final_train_rmse = final_train_mse**0.5
                final_test_rmse = final_test_mse**0.5
                mlflow.log_metric("final_train_mse", final_train_mse)
                mlflow.log_metric("final_test_mse", final_test_mse)
                mlflow.log_metric("final_train_rmse", final_train_rmse)
                mlflow.log_metric("final_test_rmse", final_test_rmse)

                # "val_label" takes precedence over "test_label" when
                # both are present.
                train_var = _label_var(dataset.get("train_label"))
                test_label = dataset.get("val_label", dataset.get("test_label"))
                test_var = _label_var(test_label)
                final_train_r2 = (
                    1.0 - final_train_mse / train_var
                    if train_var and train_var > 0
                    else float("nan")
                )
                final_test_r2 = (
                    1.0 - final_test_mse / test_var
                    if test_var and test_var > 0
                    else float("nan")
                )
                mlflow.log_metric("final_train_r2", final_train_r2)
                mlflow.log_metric("final_test_r2", final_test_r2)

                pp = {}
                if n_param:
                    final_mse_per = [float(results[f"mse_p{j}"][-1]) for j in range(n_param)]
                    pp = _per_param_metrics(final_mse_per, var_per)
                    for k, v in pp.items():
                        mlflow.log_metric(f"final_test_{k}", v)

                print(f"\nFinal Train MSE:  {final_train_mse:.6f}")
                print(f"Final Test MSE:   {final_test_mse:.6f}")
                print(f"Final Train RMSE: {final_train_rmse:.6f}")
                print(f"Final Test RMSE:  {final_test_rmse:.6f}")
                print(f"Final Train R²:   {final_train_r2:.6f}  (pooled — inflated, see below)")
                print(f"Final Test R²:    {final_test_r2:.6f}  (pooled — inflated, see below)")
                if "r2_macro" in pp:
                    names = _param_names(n_param)
                    for nm in names:
                        print(
                            f"  {nm:>3}: R²={pp[f'r2_{nm}']:.4f}  "
                            f"std-MSE={pp[f'mse_std_{nm}']:.4f}"
                        )
                    print(f"Per-param R² (honest):   {pp['r2_macro']:.4f}  (macro mean)")
                    print(f"Std-MSE (leaderboard):   {pp['mse_std_mean']:.4f}  (= 1 − R², lower=better)")

            mlflow.log_metric("training_time_sec", train_time)

        return results


_ADJECTIVES = [
    "swift",
    "bright",
    "calm",
    "bold",
    "keen",
    "warm",
    "cool",
    "fair",
    "wild",
    "deep",
    "glad",
    "pure",
    "vast",
    "free",
    "wise",
    "rare",
]
_NOUNS = [
    "fox",
    "owl",
    "elk",
    "jay",
    "ram",
    "bee",
    "ant",
    "yak",
    "emu",
    "cod",
    "hen",
    "ape",
    "bat",
    "cat",
    "dog",
    "hawk",
]

_sysrand = random.SystemRandom()


def _generate_run_name(cfg):
    adjective = _sysrand.choice(_ADJECTIVES)
    noun = _sysrand.choice(_NOUNS)
    number = _sysrand.randint(100, 999)
    return f"{adjective}-{noun}-{number}"


def _get_model_name(cfg):
    model_class = cfg.model.get("_target_", "unknown")
    return model_class.rsplit(".", 1)[-1].replace("Model", "")


def _get_dataset_name(cfg):
    dataset_class = cfg.dataset.get("_target_", "unknown")
    base_name = dataset_class.rsplit(".", 1)[-1].replace("Dataset", "")
    if base_name == "Feynman":
        return f"Feynman_{cfg.dataset.get('name', 'unknown')}"
    return base_name


def _get_shape(cfg):
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    return str(model_cfg.get("width") or model_cfg.get("layers_hidden", "unknown"))


def _flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)
