import random
import time

import numpy as np
import torch
import mlflow
from hydra.utils import instantiate
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from src import get_device

from src.training.metrics import (
    eval_metric_sums,
    score_loss_fn,
)
from src.training.plots import (
    collect_predictions,
    compute_bias_metrics,
    semantic_error_figures,
)
from src.training.utils import (
    flatten_dict,
    generate_run_name,
    get_dataset_name,
    get_model_name,
    get_shape,
)


# Versuche that train the 4-output (mu, log_sigma) head against score_loss_fn
# and get the score/coverage/mse logging path; every other objective (mse)
# gets the plain-MSE regression path.
_SCORE_OBJECTIVES = frozenset({"score", "ensemble"})


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        device_cfg = cfg.training.get("device", "auto")
        self.device = get_device() if device_cfg == "auto" else torch.device(device_cfg)
        print(f"Using device: {self.device}")
        if "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True

    def _create_loss_fn(self, objective: str):
        if objective in _SCORE_OBJECTIVES:
            # FAIR-Universe weak-lensing: minimise the negative of the
            # PDF leaderboard score with λ=1e3.  Labels must be
            # standardised by the dataset so σ lives at a natural O(1)
            # scale.  No gradient clipping / log σ clamping anymore — use a
            # small LR if the λ·MSE term destabilises training.
            return score_loss_fn
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

        # objective (configs/objective/<name>.yaml) picks both the loss
        # function and the metrics-logging path directly.
        objective = cfg.objective
        loss_fn = self._create_loss_fn(objective)
        uses_score_metrics = objective in _SCORE_OBJECTIVES

        # setup MLflow
        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment", "experiment"))
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(1)

        with mlflow.start_run(run_name=generate_run_name()):
            # log config parameters
            flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(flat_cfg)
            mlflow.log_param("parameter_count", model.parameter_count())
            mlflow.log_param("dataset", get_dataset_name(cfg))
            mlflow.log_param("model", get_model_name(cfg))
            mlflow.log_param("shape", get_shape(cfg))

            # train
            t_start = time.time()

            fit_kwargs = dict(
                dataset=dataset,
                epochs=cfg.training.epochs,
                optimizer_factory=optimizer_factory,
                loss_fn=loss_fn,
                batch_size=cfg.training.get("batch_size", -1),
                lamb=cfg.training.get("lamb", 0.0),
                num_workers=cfg.training.get("num_workers", 0),
                prefetch_factor=cfg.training.get("prefetch_factor", 4),
                grad_clip=cfg.training.get("grad_clip", None),
                early_stopping=cfg.training.get("early_stopping", False),
                es_patience=cfg.training.get("es_patience", 10),
                es_min_delta=cfg.training.get("es_min_delta", 0.0),
                es_restore_best=cfg.training.get("es_restore_best", True),
            )

            if uses_score_metrics:
                label_stats = dataset.get("label_stats")
                if label_stats is not None:
                    label_std_t = torch.as_tensor(label_stats[1], dtype=torch.float32)
                    fit_kwargs["extra_eval_metrics_fn"] = lambda p, t: eval_metric_sums(
                        p, t, label_std=label_std_t
                    )
                else:
                    fit_kwargs["extra_eval_metrics_fn"] = eval_metric_sums

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
            if uses_score_metrics:
                # ``train_loss`` / ``test_loss`` are the PDF score-loss
                # (negative leaderboard score, λ=1e3) — i.e. the exact
                # quantity submitted to Codabench.  Labels are
                # standardised, so MSE/R²/score are reported in z-space;
                # multiply MSE component-wise by std² to recover original
                # units at submission time.
                n_steps = len(results["train_loss"])
                test_var = _label_var(
                    dataset.get("val_label", dataset.get("test_label"))
                )
                # ``test_score`` is the Codabench leaderboard score — computed
                # in original Ω_m / S_8 units when labels were standardised,
                # otherwise identical to ``-score_loss``.
                score_key = (
                    "score_loss_original"
                    if "score_loss_original" in results
                    else "score_loss"
                )
                for step_i in range(n_steps):
                    tl = float(results["train_loss"][step_i])
                    vl = float(results["test_loss"][step_i])
                    test_mse = float(results["mse"][step_i])
                    metrics = {
                        "train_loss": tl,  # z-space PDF score-loss (training objective)
                        "test_loss": vl,  # z-space PDF score-loss on val
                        "test_score": -float(results[score_key][step_i]),  # Codabench
                        "test_mse": test_mse,
                        "test_rmse": test_mse**0.5,
                        "test_coverage": float(results["coverage"][step_i]),
                    }
                    if test_var and test_var > 0:
                        metrics["test_r2"] = 1.0 - test_mse / test_var
                    mlflow.log_metrics(metrics, step=step_i)

                # Final summary metrics (last epoch).
                final = {
                    k: v[-1] for k, v in results.items() if isinstance(v, list) and v
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
                    mlflow.log_metric("final_test_r2", 1.0 - final_test_mse / test_var)

                print(
                    f"\nFinal Test Score:        {final_test_score:.4f}  (Codabench scale)"
                )
                print(f"Final Test MSE:          {final_test_mse:.6f}")
                print(
                    f"Final Coverage:          {float(final['coverage']):.4f}  (target ~0.68)"
                )

                self._log_semantic_error_plots(model, dataset)
            else:
                for step_i, (tl, vl) in enumerate(
                    zip(results["train_loss"], results["test_loss"])
                ):
                    mlflow.log_metrics(
                        {
                            "train_mse": float(tl),
                            "test_mse": float(vl),
                            "train_rmse": float(tl) ** 0.5,
                            "test_rmse": float(vl) ** 0.5,
                        },
                        step=step_i,
                    )

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

                print(f"\nFinal Train MSE:  {final_train_mse:.6f}")
                print(f"Final Test MSE:   {final_test_mse:.6f}")
                print(f"Final Train RMSE: {final_train_rmse:.6f}")
                print(f"Final Test RMSE:  {final_test_rmse:.6f}")
                print(f"Final Train R²:   {final_train_r2:.6f}")
                print(f"Final Test R²:    {final_test_r2:.6f}")

            mlflow.log_metric("training_time_sec", train_time)

        return results

    def _log_semantic_error_plots(self, model, dataset):
        val_input = dataset.get("val_input")
        val_label = dataset.get("val_label")
        label_stats = dataset.get("label_stats")
        if val_input is None or val_label is None or label_stats is None:
            return

        target_names = list(self.cfg.dataset.get("target_names", ["Omega_m", "S8"]))
        y_true, y_pred = collect_predictions(model, val_input, val_label, self.device)

        label_mean, label_std = label_stats
        y_true = y_true * label_std + label_mean
        y_pred = y_pred * label_std + label_mean

        for key, value in compute_bias_metrics(y_true, y_pred, target_names).items():
            mlflow.log_metric(f"final_test_{key}", value)

        for name, fig in semantic_error_figures(y_true, y_pred, target_names).items():
            mlflow.log_figure(fig, f"semantic_checks/{name}.png")
            plt.close(fig)
