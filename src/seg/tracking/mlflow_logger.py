# src/ods/tracking/mlflow_logger.py
import mlflow
from pathlib import Path


class MLflowLogger:
    def __init__(self, tracking_uri: str, experiment_name: str, run_name: str | None = None):
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"[MLflow] Run ID: {self.run_id}")

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def finish(self):
        mlflow.end_run()