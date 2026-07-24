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
    plot_histogram_pred_vs_groundtruth,
    plot_predicted_vs_groundtruth,
)
from src.training.utils import (
    flatten_dict,
    generate_run_name,
    get_dataset_name,
    get_model_name,
    get_shape,
)

# objectives with a 4-output (mu, log_sigma) head + score/coverage logging; else plain MSE
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
            # negative FAIR-Universe score, λ=1e3; labels must be standardised (see dataset)
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

        self._seed_everything(cfg.get("seed", 42))

        dataset_obj = instantiate(cfg.dataset)
        dataset = dataset_obj.create()

        model = instantiate(cfg.model)
        model.build(device=self.device)

        optimizer_factory = instantiate(cfg.optimizer)

        objective = cfg.objective
        loss_fn = self._create_loss_fn(objective)
        uses_score_metrics = objective in _SCORE_OBJECTIVES

        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment", "experiment"))
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(1)

        with mlflow.start_run(run_name=generate_run_name()):
            flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(flat_cfg)
            mlflow.log_param("parameter_count", model.parameter_count())
            mlflow.log_param("dataset", get_dataset_name(cfg))
            mlflow.log_param("model", get_model_name(cfg))
            mlflow.log_param("shape", get_shape(cfg))

            t_start = time.time()

            fit_kwargs = dict(
                dataset=dataset,
                epochs=cfg.training.epochs,
                optimizer_factory=optimizer_factory,
                loss_fn=loss_fn,
                batch_size=cfg.training.batch_size,
                lamb=cfg.training.get("lamb", 0.0),
                num_workers=cfg.training.get("num_workers", 0),
                prefetch_factor=cfg.training.get("prefetch_factor", 4),
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

            if uses_score_metrics:
                # labels are standardised, so MSE/R²/score below are logged in z-space
                n_steps = len(results["train_loss"])
                test_var = float(np.var(dataset.get("val_label")))
                # score_loss_original (if present) recovers the Codabench score in raw Om/S8 units
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

                train_var = float(np.var(dataset.get("train_label")))
                test_var = float(np.var(dataset.get("val_label")))
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

            val_input = dataset.get("val_input")
            val_label = dataset.get("val_label")
            label_stats = dataset.get("label_stats")
            if (
                val_input is not None
                and val_label is not None
                and label_stats is not None
            ):
                y_true, y_pred = collect_predictions(
                    model, val_input, val_label, self.device
                )
                label_mean, label_std = label_stats
                y_true = y_true * label_std + label_mean
                y_pred = y_pred * label_std + label_mean

                target_names = list(cfg.dataset.get("target_names", ["Omega_m", "S8"]))
                # purely diagnostic: don't let a plotting failure kill the whole Optuna sweep
                try:
                    figs = {
                        "histogram_pred_vs_groundtruth": plot_histogram_pred_vs_groundtruth(
                            y_true, y_pred, target_names
                        ),
                        "predicted_vs_groundtruth": plot_predicted_vs_groundtruth(
                            y_true, y_pred, target_names
                        ),
                    }
                    for name, fig in figs.items():
                        mlflow.log_figure(fig, f"semantic_checks/{name}.png")
                        plt.close(fig)
                except Exception:
                    import traceback

                    traceback.print_exc()
                    print("semantic-check plotting failed -> skipping, trial continues")

            mlflow.log_metric("training_time_sec", train_time)

        return results
