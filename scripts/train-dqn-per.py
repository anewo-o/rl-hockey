import os
import sys
import yaml
import ale_py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from src.env_utils import create_ice_hockey_env
from src.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.per_dqn import PERDQN

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(resume_path=None):
    models_dir = "models"
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    logs_dir = "logs"
    config_path = os.path.join("configs", "dqn_atari.yml")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Chargement de la configuration depuis {config_path}...")
    config = load_config(config_path)
    env_cfg = config["environment"]
    dqn_cfg = config["dqn"]
    train_cfg = config["training"]


    print("Initialisation des environnements d'entraînement...")
    train_env = create_ice_hockey_env(
        n_envs=env_cfg["n_envs"],
        render_mode=None,
        seed=0
    )


    print("Initialisation de l'environnement d'évaluation...")
    eval_env = create_ice_hockey_env(
        n_envs=1,
        render_mode=None,
        seed=42
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=max(1, train_cfg["eval_freq"] // env_cfg["n_envs"]),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint_freq"] // env_cfg["n_envs"],
        save_path=checkpoints_dir,
        name_prefix=train_cfg["per_model_name"]
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Chargement
    if resume_path and os.path.exists(resume_path):
        print(f"---- REPRISE DE L'ENTRAÎNEMENT DEPUIS {resume_path} ----")
        model = DQN.load(
            resume_path,
            env=train_env,
            verbose=1,
            tensorboard_log=logs_dir,
        )
        reset_timesteps = False
    else:
        print("Nouveau modèle DQN")
        
        model = PERDQN(
            env=train_env,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs=dict(alpha=0.6, beta=0.4),
            verbose=1,
            tensorboard_log=logs_dir,
            device="auto",
            **dqn_cfg
        )
        reset_timesteps = True

    print("Début de l'entraînement...")
    try:
        
        # Entraînement personnalisé pour intégrer PER
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
            tb_log_name=train_cfg["per_model_name"],
            reset_num_timesteps=reset_timesteps,
        )
        
    except KeyboardInterrupt:
        print("\nInterruption, sauvegarde du modèle...")

    finally:
        final_model_path = os.path.join(models_dir, f"{train_cfg['per_model_name']}_final")
        model.save(final_model_path)
        print(f"Modèle final sauvegardé sous {final_model_path}")
        print(f"Le MEILLEUR modèle a été sauvegardé automatiquement sous {models_dir}/best_model.zip")

        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    train()