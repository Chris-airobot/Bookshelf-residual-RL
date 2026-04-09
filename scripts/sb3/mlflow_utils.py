"""Minimal MLflow helpers for SB3 training scripts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


def is_mlflow_available() -> bool:
    return mlflow is not None


def maybe_start_mlflow_run(
    enabled: bool,
    run_name: str,
    tracking_uri: str | None,
    experiment_name: str,
    tags: dict[str, str] | None = None,
) -> bool:
    """Start an MLflow run if available and requested."""
    if not enabled or mlflow is None:
        return False

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name, tags=tags or {})

    # Optional API on newer MLflow versions; guard for compatibility.
    if hasattr(mlflow, "enable_system_metrics_logging"):
        try:
            mlflow.enable_system_metrics_logging()
        except Exception:
            pass
    return True


def end_mlflow_run_if_active() -> None:
    if mlflow is None:
        return
    try:
        mlflow.end_run()
    except Exception:
        pass


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def log_config_snapshot(env_cfg: Any, agent_cfg: dict[str, Any], args_dict: dict[str, Any]) -> None:
    """Log compact config-level parameters into MLflow params."""
    if mlflow is None:
        return

    params: dict[str, Any] = {
        "task": args_dict.get("task"),
        "seed": agent_cfg.get("seed"),
        "num_envs": getattr(getattr(env_cfg, "scene", None), "num_envs", None),
        "device": args_dict.get("device"),
        "video": args_dict.get("video"),
        "sb3_policy": agent_cfg.get("policy"),
        "sb3_n_steps": agent_cfg.get("n_steps"),
        "sb3_batch_size": agent_cfg.get("batch_size"),
        "sb3_gamma": agent_cfg.get("gamma"),
        "sb3_learning_rate": agent_cfg.get("learning_rate"),
        "sb3_n_epochs": agent_cfg.get("n_epochs"),
        "sb3_clip_range": agent_cfg.get("clip_range"),
        "sb3_ent_coef": agent_cfg.get("ent_coef"),
    }
    clean_params = {k: _to_jsonable(v) for k, v in params.items() if v is not None}
    mlflow.log_params(clean_params)


def log_artifact_if_exists(path: str | os.PathLike[str], artifact_path: str | None = None) -> None:
    if mlflow is None:
        return
    if Path(path).exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_text_artifact(text: str, filename: str, artifact_path: str | None = None) -> None:
    if mlflow is None:
        return
    mlflow.log_text(text, artifact_file=f"{artifact_path}/{filename}" if artifact_path else filename)


def log_json_artifact(payload: dict[str, Any], filename: str, artifact_path: str | None = None) -> None:
    if mlflow is None:
        return
    mlflow.log_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True),
        artifact_file=f"{artifact_path}/{filename}" if artifact_path else filename,
    )


class MlflowSb3MetricsCallback(BaseCallback):
    """Periodically mirrors SB3 logger scalars into MLflow metrics."""

    def __init__(self, log_every_n_calls: int = 200):
        super().__init__(verbose=0)
        self.log_every_n_calls = max(1, int(log_every_n_calls))

    def _on_step(self) -> bool:
        if mlflow is None or (self.n_calls % self.log_every_n_calls != 0):
            return True

        logger_values = getattr(self.model.logger, "name_to_value", {})
        for name, value in logger_values.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mlflow.log_metric(name, float(value), step=int(self.num_timesteps))
        return True
