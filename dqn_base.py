"""Train DQN baseline on ALE/IceHockey-v5 with rich experiment logging.

Examples:
    C:\venv311rl\Scripts\python.exe dqn_train.py --smoke
    C:\venv311rl\Scripts\python.exe dqn_train.py
    C:\venv311rl\Scripts\python.exe dqn_train.py --full
"""

import argparse
import json
import os
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# ─────────────────────────────────────────────
# 1. Enregistrement ALE (obligatoire, Gymnasium >= 1.0)
# ─────────────────────────────────────────────
gym.register_envs(ale_py)


class TrainingMetricsCallback(BaseCallback):
    """Persist periodic training metrics for cross-algorithm comparison."""

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
        mean_ep_reward_100 = None
        mean_ep_length_100 = None
        if ep_infos:
            rewards = [float(ep.get("r", 0.0)) for ep in ep_infos]
            lengths = [float(ep.get("l", 0.0)) for ep in ep_infos]
            mean_ep_reward_100 = float(np.mean(rewards))
            mean_ep_length_100 = float(np.mean(lengths))

        logger_values = dict(getattr(self.model.logger, "name_to_value", {}))
        row = {
            "timesteps": int(self.model.num_timesteps),
            "exploration_rate": float(getattr(self.model, "exploration_rate", np.nan)),
            "mean_ep_reward_100": mean_ep_reward_100,
            "mean_ep_length_100": mean_ep_length_100,
            "train_loss": _safe_float(logger_values.get("train/loss")),
            "learning_rate": _safe_float(logger_values.get("train/learning_rate")),
            "n_updates": _safe_float(logger_values.get("train/n_updates")),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        fps = self.model.num_timesteps / max(elapsed, 1e-8)

        ep_infos = list(self.model.ep_info_buffer)
        final_mean_ep_reward_100 = None
        final_mean_ep_length_100 = None
        if ep_infos:
            final_mean_ep_reward_100 = float(np.mean([float(ep.get("r", 0.0)) for ep in ep_infos]))
            final_mean_ep_length_100 = float(np.mean([float(ep.get("l", 0.0)) for ep in ep_infos]))

        summary = {
            "algorithm": "DQN",
            "env_id": ENV_ID,
            "total_timesteps": int(self.model.num_timesteps),
            "duration_seconds": round(elapsed, 2),
            "fps_estimate": round(fps, 2),
            "final_mean_ep_reward_100": final_mean_ep_reward_100,
            "final_mean_ep_length_100": final_mean_ep_length_100,
            "metrics_file": str(self.metrics_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _compute_auc(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    return float(np.trapezoid(y, x))


def _first_timestep_reaching(timesteps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    if timesteps.size == 0 or values.size == 0:
        return None
    idx = np.where(values >= threshold)[0]
    if idx.size == 0:
        return None
    return int(timesteps[idx[0]])


def build_report_artifacts(logs_dir: Path) -> dict:
    """Create report-ready metrics and figures from training logs."""
    figures_dir = logs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = logs_dir / "dqn_base_metrics.jsonl"
    rows = _load_jsonl(metrics_jsonl)

    train_ts = np.array([r["timesteps"] for r in rows], dtype=np.float64) if rows else np.array([])
    train_reward = np.array(
        [r["mean_ep_reward_100"] if r["mean_ep_reward_100"] is not None else np.nan for r in rows],
        dtype=np.float64,
    ) if rows else np.array([])
    eps_rate = np.array([r["exploration_rate"] for r in rows], dtype=np.float64) if rows else np.array([])

    eval_npz = logs_dir / "evaluations.npz"
    eval_ts = np.array([])
    eval_mean = np.array([])
    eval_std = np.array([])
    if eval_npz.exists():
        data = np.load(eval_npz)
        eval_ts = data["timesteps"].astype(np.float64)
        eval_results = data["results"].astype(np.float64)
        eval_mean = eval_results.mean(axis=1)
        eval_std = eval_results.std(axis=1)

    if train_ts.size > 0:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(train_ts, train_reward, label="mean_ep_reward_100", color="#1f77b4")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Train reward (moving avg)")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(train_ts, eps_rate, label="exploration_rate", color="#ff7f0e", linestyle="--")
        ax2.set_ylabel("Exploration rate")

        fig.suptitle("DQN Training Dynamics - IceHockey")
        fig.tight_layout()
        fig.savefig(figures_dir / "dqn_base_training_dynamics.png", dpi=160)
        plt.close(fig)

    if eval_ts.size > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_ts, eval_mean, label="mean_eval_reward", color="#2ca02c")
        ax.fill_between(eval_ts, eval_mean - eval_std, eval_mean + eval_std, alpha=0.2, color="#2ca02c")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Eval reward")
        ax.grid(alpha=0.25)
        ax.set_title("DQN Evaluation Curve (mean +/- std)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "dqn_base_eval_curve.png", dpi=160)
        plt.close(fig)

    auc_eval = _compute_auc(eval_ts, eval_mean) if eval_ts.size > 0 else None
    best_eval_idx = int(np.nanargmax(eval_mean)) if eval_mean.size > 0 else None

    report = {
        "algorithm": "DQN",
        "env_id": ENV_ID,
        "train_curve_points": int(train_ts.size),
        "eval_curve_points": int(eval_ts.size),
        "best_eval_mean_reward": float(eval_mean[best_eval_idx]) if best_eval_idx is not None else None,
        "best_eval_std_reward": float(eval_std[best_eval_idx]) if best_eval_idx is not None else None,
        "best_eval_timestep": int(eval_ts[best_eval_idx]) if best_eval_idx is not None else None,
        "last_eval_mean_reward": float(eval_mean[-1]) if eval_mean.size > 0 else None,
        "last_eval_std_reward": float(eval_std[-1]) if eval_std.size > 0 else None,
        "auc_eval_reward_timesteps": auc_eval,
        "timesteps_to_reach_reward_0": _first_timestep_reaching(eval_ts, eval_mean, 0.0),
        "timesteps_to_reach_reward_1": _first_timestep_reaching(eval_ts, eval_mean, 1.0),
        "figures": {
            "training_dynamics": str(figures_dir / "dqn_base_training_dynamics.png"),
            "eval_curve": str(figures_dir / "dqn_base_eval_curve.png"),
        },
    }

    out = logs_dir / "dqn_base_report_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN baseline training for ALE/IceHockey-v5")
    parser.add_argument("--smoke", action="store_true", help="Short run for sanity check (20k steps)")
    parser.add_argument("--full", action="store_true", help="Long run (1M steps)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes per evaluation")
    parser.add_argument("--render", action="store_true", help="Render at end of training")
    parser.add_argument("--metrics-freq", type=int, default=2_000, help="Frequency for custom metrics logging")
    return parser.parse_args()


def resolve_total_steps(args: argparse.Namespace) -> int:
    if args.timesteps is not None:
        return args.timesteps
    if args.full:
        return 1_000_000
    if args.smoke:
        return 20_000
    return 100_000

# ─────────────────────────────────────────────
# 2. Créer l'environnement vectorisé + frame stacking
#    make_atari_env s'occupe de : NoopReset, MaxAndSkip,
#    EpisodicLife, FireReset, WarpFrame (84x84 gris), ScaledFloat
# ─────────────────────────────────────────────
ENV_ID = "ALE/IceHockey-v5"
N_ENVS = 1
N_STACK = 4


def main() -> None:
    args = parse_args()
    total_steps = resolve_total_steps(args)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    vec_env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=42, monitor_dir="logs/monitor_train")
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)

    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=123, monitor_dir="logs/monitor_eval")
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
    eval_env = VecTransposeImage(eval_env)

# ─────────────────────────────────────────────
# 3. Callback d'évaluation automatique
# ─────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    metrics_callback = TrainingMetricsCallback(
        metrics_path=Path("logs/dqn_base_metrics.jsonl"),
        summary_path=Path("logs/dqn_base_summary.json"),
        log_freq=args.metrics_freq,
    )

    callbacks = CallbackList([eval_callback, metrics_callback])

# ─────────────────────────────────────────────
# 4. Créer le modèle DQN
#    policy="CnnPolicy" = réseau convolutif (Atari standard)
# ─────────────────────────────────────────────
    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        optimize_memory_usage=False,
        verbose=1,
    )

    print(f"\n>>> Démarrage entraînement DQN sur {ENV_ID}")
    print(f"    Steps totaux : {total_steps}")
    print(f"    ε : 1.0 -> 0.05 (sur {int(total_steps * 0.1)} steps)")
    print("    Logs custom : logs/dqn_base_metrics.jsonl")
    print("    Résumé : logs/dqn_base_summary.json\n")

# ─────────────────────────────────────────────
# 5. Entraînement
# ─────────────────────────────────────────────
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        log_interval=10,
    )

# ─────────────────────────────────────────────
# 6. Sauvegarder le modèle final
# ─────────────────────────────────────────────
    model.save("models/dqn_icehockey_final")
    print("\n>>> Modèle sauvegardé dans models/dqn_icehockey_final.zip")

    report = build_report_artifacts(Path("logs"))
    print("    Rapport: logs/dqn_base_report_metrics.json")
    print("    Figures: logs/figures/dqn_base_training_dynamics.png, logs/figures/dqn_base_eval_curve.png")
    if report.get("best_eval_mean_reward") is not None:
        print(f"    Best eval mean reward: {report['best_eval_mean_reward']:.3f}")

# ─────────────────────────────────────────────
# 7. Évaluation du modèle entraîné (rendu visuel)
# ─────────────────────────────────────────────
    if args.render:
        print("\n>>> Évaluation du modèle (fenêtre visuelle)...")
        render_env = make_atari_env(ENV_ID, n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
        render_env = VecFrameStack(render_env, n_stack=N_STACK)
        render_env = VecTransposeImage(render_env)

        obs = render_env.reset()
        total_reward = 0
        n_episodes = 0

        for _ in range(3000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = render_env.step(action)
            total_reward += reward[0]
            if done[0]:
                n_episodes += 1
                print(f"  Épisode {n_episodes} terminé | reward cumulé = {total_reward:.1f}")
                total_reward = 0
                if n_episodes >= 3:
                    break

        render_env.close()

    vec_env.close()
    eval_env.close()
    print("\n>>> Terminé.")


if __name__ == "__main__":
    main()
