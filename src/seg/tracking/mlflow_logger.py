"""
src/seg/tracking/mlflow_logger.py
-----------------------------------
Thin wrapper around MLflow for experiment tracking.

Creates or resumes an MLflow run and exposes simple log_* methods
so the Trainer doesn't need to import mlflow directly.

Usage:
    mlf = MLflowLogger(
        tracking_uri     = "outputs/mlruns",
        experiment_name  = "deeplabv3plus_cityscapes",
        run_name         = "exp01_resnet50",
    )
    mlf.log_params({"lr": 0.007, "backbone": "resnet50"})
    mlf.log_metrics({"mIoU": 0.43, "val_loss": 0.72}, step=10)
    mlf.log_artifact("outputs/plots/class_iou.png")
    mlf.finish()
"""

import mlflow
from pathlib import Path


class MLflowLogger:
    """
    Wraps MLflow tracking for a single experiment run.

    Args:
        tracking_uri    : local path or remote URI for the MLflow backend
        experiment_name : name of the MLflow experiment
        run_name        : display name for this run (usually the exp ID)
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
    ):
        # Create the local directory if using a file-based backend
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"[MLflow] Experiment: {experiment_name}  |  Run ID: {self.run_id}")

    def log_params(self, params: dict):
        """
        Log hyperparameters (called once at training start).
        Values are stringified automatically by MLflow.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int):
        """
        Log scalar metrics for a training step / epoch.

        Args:
            metrics : dict of {metric_name: float_value}
            step    : epoch or global step number
        """
        # MLflow only accepts numeric values; skip non-floats silently
        numeric = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, local_path: str):
        """
        Upload a file (image, plot, config) as an artifact for this run.

        Args:
            local_path : path to the file on disk
        """
        mlflow.log_artifact(local_path)

    def finish(self):
        """End the MLflow run. Call once when training is complete."""
        mlflow.end_run()
        print(f"[MLflow] Run {self.run_id} finished.")