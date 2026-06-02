"""CSV episode metrics callback for SB3 bookshelf training."""

import csv
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


_FAILURE_REASONS = {
    0: "none",
    1: "success",
    2: "drop",
    3: "timeout",
    4: "not_push",
    5: "depth",
    6: "lateral",
    7: "z",
    8: "yaw",
    9: "upright",
    10: "unstable",
    11: "oob",
    12: "fell",
}

_MODE_NAMES = {
    0: "insert",
    1: "scripted",
    2: "push",
}


def _scalar(value, env_idx: int | None = None, default=None):
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if env_idx is not None and hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
            value = value[env_idx]
        if hasattr(value, "item"):
            return value.item()
        return value
    except Exception:
        return default


def _metrics_from_infos(infos, env_idx: int):
    if isinstance(infos, dict):
        metrics = infos.get("episode_metrics")
        if isinstance(metrics, dict):
            return {key: _scalar(value, env_idx) for key, value in metrics.items()}
        return {
            key.removeprefix("episode_metric_"): _scalar(value, env_idx)
            for key, value in infos.items()
            if key.startswith("episode_metric_")
        }
    if isinstance(infos, (list, tuple)) and env_idx < len(infos) and isinstance(infos[env_idx], dict):
        info = infos[env_idx]
        metrics = info.get("episode_metrics")
        if isinstance(metrics, dict):
            return {key: _scalar(value) for key, value in metrics.items()}
        return {
            key.removeprefix("episode_metric_"): _scalar(value)
            for key, value in info.items()
            if key.startswith("episode_metric_")
        }
    return {}


class EpisodeMetricsCsvCallback(BaseCallback):
    """Write compact per-episode training metrics and rolling summaries to CSV."""

    episode_fields = [
        "global_step",
        "episode_id",
        "env_id",
        "slot_center_y",
        "slot_clearance",
        "missing_book_index",
        "success",
        "failure_reason",
        "episode_length",
        "episode_return",
        "final_lat_err",
        "final_z_err",
        "final_yaw_err_deg",
        "final_rear_to_mouth",
        "final_front_to_back",
        "release_step",
        "push_steps",
        "mode_at_done",
    ]
    summary_fields = [
        "global_step",
        "episodes",
        "window_size",
        "mean_slot_center_y",
        "mean_slot_clearance",
        "success_rate",
        "timeout_rate",
        "drop_rate",
        "mean_episode_return",
        "mean_episode_length",
        "mean_final_lat_err",
        "mean_final_z_err",
        "mean_final_yaw_err_deg",
        "mean_final_rear_to_mouth",
        "mean_final_front_to_back",
    ]

    def __init__(
        self,
        log_dir: str,
        window_size: int = 1000,
        summary_every_episodes: int = 1000,
        flush_every_episodes: int = 100,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.window_size = int(window_size)
        self.summary_every_episodes = int(summary_every_episodes)
        self.flush_every_episodes = int(flush_every_episodes)
        self.episode_count = 0
        self._returns = None
        self._lengths = None
        self._window = deque(maxlen=self.window_size)
        self._episode_file = None
        self._summary_file = None
        self._episode_writer = None
        self._summary_writer = None

    def _on_training_start(self) -> None:
        num_envs = int(self.training_env.num_envs)
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_file = (self.log_dir / "episode_metrics.csv").open("w", newline="")
        self._summary_file = (self.log_dir / "training_summary.csv").open("w", newline="")
        self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=self.episode_fields)
        self._summary_writer = csv.DictWriter(self._summary_file, fieldnames=self.summary_fields)
        self._episode_writer.writeheader()
        self._summary_writer.writeheader()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])
        if self._returns is None or self._lengths is None or rewards.size == 0:
            return True

        self._returns[: rewards.size] += rewards
        self._lengths[: rewards.size] += 1

        for env_idx, done in enumerate(dones):
            if not done:
                continue
            metrics = _metrics_from_infos(infos, env_idx)
            failure_code = int(metrics.get("failure_code", 0) or 0)
            mode_value = metrics.get("mode_at_done", None)
            mode_code = -1 if mode_value is None or mode_value == "" else int(mode_value)
            row = {
                "global_step": int(self.num_timesteps),
                "episode_id": self.episode_count,
                "env_id": env_idx,
                "slot_center_y": metrics.get("slot_center_y", ""),
                "slot_clearance": metrics.get("slot_clearance", ""),
                "missing_book_index": metrics.get("missing_book_index", ""),
                "success": int(bool(metrics.get("success", failure_code == 1))),
                "failure_reason": _FAILURE_REASONS.get(failure_code, f"code_{failure_code}"),
                "episode_length": int(self._lengths[env_idx]),
                "episode_return": float(self._returns[env_idx]),
                "final_lat_err": metrics.get("final_lat_err", ""),
                "final_z_err": metrics.get("final_z_err", ""),
                "final_yaw_err_deg": metrics.get("final_yaw_err_deg", ""),
                "final_rear_to_mouth": metrics.get("final_rear_to_mouth", ""),
                "final_front_to_back": metrics.get("final_front_to_back", ""),
                "release_step": metrics.get("release_step", ""),
                "push_steps": metrics.get("push_steps", ""),
                "mode_at_done": _MODE_NAMES.get(mode_code, ""),
            }
            self._episode_writer.writerow(row)
            self._window.append(row)
            self.episode_count += 1
            self._returns[env_idx] = 0.0
            self._lengths[env_idx] = 0

            if self.episode_count % self.summary_every_episodes == 0:
                self._write_summary()
                self._episode_file.flush()
                self._summary_file.flush()
            elif self.episode_count % self.flush_every_episodes == 0:
                self._episode_file.flush()

        return True

    def _mean_numeric(self, rows, key: str):
        vals = []
        for row in rows:
            value = row.get(key, "")
            if value == "":
                continue
            vals.append(float(value))
        return float(np.mean(vals)) if vals else ""

    def _write_summary(self) -> None:
        rows = list(self._window)
        if not rows:
            return
        n = len(rows)
        summary = {
            "global_step": int(self.num_timesteps),
            "episodes": self.episode_count,
            "window_size": n,
            "mean_slot_center_y": self._mean_numeric(rows, "slot_center_y"),
            "mean_slot_clearance": self._mean_numeric(rows, "slot_clearance"),
            "success_rate": float(np.mean([int(row["success"]) for row in rows])),
            "timeout_rate": float(np.mean([row["failure_reason"] == "timeout" for row in rows])),
            "drop_rate": float(np.mean([row["failure_reason"] == "drop" for row in rows])),
            "mean_episode_return": self._mean_numeric(rows, "episode_return"),
            "mean_episode_length": self._mean_numeric(rows, "episode_length"),
            "mean_final_lat_err": self._mean_numeric(rows, "final_lat_err"),
            "mean_final_z_err": self._mean_numeric(rows, "final_z_err"),
            "mean_final_yaw_err_deg": self._mean_numeric(rows, "final_yaw_err_deg"),
            "mean_final_rear_to_mouth": self._mean_numeric(rows, "final_rear_to_mouth"),
            "mean_final_front_to_back": self._mean_numeric(rows, "final_front_to_back"),
        }
        self._summary_writer.writerow(summary)

    def _on_training_end(self) -> None:
        if self._summary_writer is not None:
            self._write_summary()
        for file_obj in (self._episode_file, self._summary_file):
            if file_obj is not None:
                file_obj.flush()
                file_obj.close()
