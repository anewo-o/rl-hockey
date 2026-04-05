"""Train DQN with OCAtari + RAM features on ALE/IceHockey-v5."""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from ocatari.core import OCAtari
from wrappers import IceHockeyShapingWrapper


class OCAtariIceHockeyEnv(gym.Env):
    """Gym env wrapper for OCAtari Ice Hockey with RAM features."""

    MAX_OBJECTS = 12
    FEATURES_PER_OBJECT = 11  # rgb(3), xy(2), prev_xy(2), wh(2), visible, hud

    def __init__(self):
        base_env = OCAtari("ALE/IceHockey-v5", mode="ram")
        self.env = IceHockeyShapingWrapper(base_env)
        # Observation: the first 12 OCAtari objects with rich per-object features.
        obs_dim = self.MAX_OBJECTS * self.FEATURES_PER_OBJECT
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_dim,), dtype=np.float32)
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        features = self._extract_features()
        return features, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action) 
        features = self._extract_features()
        return features, reward, done, truncated, info

    def _extract_features(self):
        objects = list(self.env.unwrapped.objects)
        features = []

        for index in range(self.MAX_OBJECTS):
            if index < len(objects):
                features.extend(self._object_to_features(objects[index]))
            else:
                features.extend([0.0] * self.FEATURES_PER_OBJECT)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _object_to_features(obj):
        rgb = getattr(obj, "_rgb", (0, 0, 0))
        xy = getattr(obj, "_xy", (getattr(obj, "x", 0.0), getattr(obj, "y", 0.0)))
        prev_xy = getattr(obj, "_prev_xy", None)
        prev_x = float(prev_xy[0]) if prev_xy else 0.0
        prev_y = float(prev_xy[1]) if prev_xy else 0.0
        wh = getattr(obj, "wh", (0, 0))
        w = float(wh[0]) if wh else 0.0
        h = float(wh[1]) if wh else 0.0
        visible = 1.0 if bool(getattr(obj, "_visible", True)) else 0.0
        hud = 1.0 if bool(getattr(obj, "hud", False)) else 0.0

        return [
            float(rgb[0]) if len(rgb) > 0 else 0.0,
            float(rgb[1]) if len(rgb) > 1 else 0.0,
            float(rgb[2]) if len(rgb) > 2 else 0.0,
            float(xy[0]) if len(xy) > 0 else 0.0,
            float(xy[1]) if len(xy) > 1 else 0.0,
            prev_x,
            prev_y,
            w,
            h,
            visible,
            hud,
        ]

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class TrainingMetricsCallback(BaseCallback):
    """Persist periodic training metrics."""

    def __init__(self, metrics_path: Path, summary_path: Path, log_freq: int = 2_000):
        super().__init__(verbose=0)
        self.metrics_path = metrics_path
        self.summary_path = summary_path
        self.log_freq = log_freq
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.log_freq != 0:
            return True

        ep_infos = list(self.model.ep_info_buffer)
        logger_values = dict(getattr(self.model.logger, "name_to_value", {}))
        mean_ep_reward_100 = None
        if ep_infos:
            rewards = [float(ep.get("r", 0.0)) for ep in ep_infos]
            mean_ep_reward_100 = float(np.mean(rewards))

        row = {
            "timesteps": int(self.model.num_timesteps),
            "exploration_rate": float(getattr(self.model, "exploration_rate", np.nan)),
            "mean_ep_reward_100": mean_ep_reward_100,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "mean_base_reward": _safe_float(logger_values.get("rollout/ep_base_rew_mean")),
            "mean_shaping_reward": _safe_float(logger_values.get("rollout/ep_shaping_rew_mean")),
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        fps = self.model.num_timesteps / max(elapsed, 1e-8)
        logger_values = dict(getattr(self.model.logger, "name_to_value", {}))

        summary = {
            "algorithm": "DQN_OCAtari",
            "total_timesteps": int(self.model.num_timesteps),
            "duration_seconds": round(elapsed, 2),
            "fps_estimate": round(fps, 2),
            "mean_base_reward": _safe_float(logger_values.get("rollout/ep_base_rew_mean")),
            "mean_shaping_reward": _safe_float(logger_values.get("rollout/ep_shaping_rew_mean")),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN OCAtari training")
    parser.add_argument("--smoke", action="store_true", help="Short run (20k steps)")
    parser.add_argument("--full", action="store_true", help="Long run (1M steps)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    return parser.parse_args()


def resolve_total_steps(args: argparse.Namespace) -> int:
    if args.timesteps is not None:
        return args.timesteps
    if args.full:
        return 1_000_000
    if args.smoke:
        return 20_000
    return 100_000


def main() -> None:
    args = parse_args()
    total_steps = resolve_total_steps(args)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    vec_env = make_vec_env(lambda: OCAtariIceHockeyEnv(), n_envs=1, seed=42)
    eval_env = make_vec_env(lambda: OCAtariIceHockeyEnv(), n_envs=1, seed=123)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    metrics_callback = TrainingMetricsCallback(
        metrics_path=Path("logs/dqn_ocatari_metrics.jsonl"),
        summary_path=Path("logs/dqn_ocatari_summary.json"),
        log_freq=2_000,
    )

    callbacks = CallbackList([eval_callback, metrics_callback])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
    )

    print(f"\n>>> Démarrage DQN OCAtari, steps: {total_steps}")

    model.learn(total_timesteps=total_steps, callback=callbacks)

    model.save("models/dqn_ocatari_final")
    print("Modèle sauvegardé")


if __name__ == "__main__":
    main()