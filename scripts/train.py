import os
import sys
import yaml
import argparse
from typing import Callable, Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from src.env_utils import create_ice_hockey_env
from src.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.per_dqn import PERDQN
from src.ppo_addendum import PartialPPO


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Crée une fonction qui diminue linéairement le taux d'apprentissage.
    
    Args:
        initial_value (float): Le taux d'apprentissage de départ (ex: 0.00025)
        
    Returns:
        Callable: Une fonction appelée par PPO à chaque mise à jour.
    """
    def func(progress_remaining: float) -> float:
        """
        Calcule le taux actuel.
        `progress_remaining` commence à 1.0 et descend jusqu'à 0.0
        """
        # ex: s'il reste 50% de l'entraînement, le lr sera de 0.5 * initial_value
        return progress_remaining * initial_value
    
    return func


def load_config(algo: str) -> Dict[str, Any]:
    """Charge le bon fichier de config selon l'algo."""
    config_name = "ppo_atari.yml" if "ppo" in algo.lower() else "dqn_atari.yml"
    config_path = os.path.join("configs", config_name)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_model_class(algo_name: str):
    """Mappe le nom de l'algo à sa classe et ses paramètres spécifiques."""
    mapping = {
        "ppo": (PPO, {}),
        "partial-ppo": (PartialPPO, {}), # Le partial_horizon sera extrait de la config
        "dqn": (DQN, {}),
        "per-dqn": (PERDQN, {
            "replay_buffer_class": PrioritizedReplayBuffer,
            "replay_buffer_kwargs": dict(alpha=0.6, beta=0.4)
        })
    }
    if algo_name not in mapping:
        raise ValueError(f"Algo {algo_name} non supporté. Choix : {list(mapping.keys())}")
    return mapping[algo_name]


def train(algo_name: str, resume_path: str = None):
    # setup des dossiers
    models_dir, logs_dir = "models", "logs"
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    for d in [models_dir, checkpoints_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # Chargement config et classes
    config = load_config(algo_name)
    env_cfg = config["environment"]
    train_cfg = config["training"]
    algo_cfg = config.get("ppo") if "ppo" in algo_name else config.get("dqn")
    
    model_class, extra_kwargs = get_model_class(algo_name)
    model_name = train_cfg["model_name"]

    # Gestion spécifique des hyperparamètres
    if "ppo" in algo_name:
        initial_lr = algo_cfg["learning_rate"]
        algo_cfg["learning_rate"] = linear_schedule(initial_lr)
        if algo_name == "partial-ppo":
            extra_kwargs["partial_horizon"] = int(algo_cfg["n_steps"] / 2)

    # Environnements
    print(f"--- Initialisation des environnements pour {algo_name} ---")
    train_env = create_ice_hockey_env(n_envs=env_cfg["n_envs"], seed=0)
    eval_env = create_ice_hockey_env(n_envs=1, seed=42)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=max(1, train_cfg["eval_freq"] // env_cfg["n_envs"]),
        n_eval_episodes=5,
        deterministic=True
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint_freq"] // env_cfg["n_envs"],
        save_path=checkpoints_dir,
        name_prefix=model_name
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Initialisation ou Reprise
    if resume_path and os.path.exists(resume_path):
        print(f"---- REPRISE DE L'ENTRAINEMENT : {resume_path} ----")
        load_kwargs = {"env": train_env, "tensorboard_log": logs_dir}
        if "ppo" in algo_name:
            load_kwargs["custom_objects"] = {"learning_rate": algo_cfg["learning_rate"]}
        
        model = model_class.load(resume_path, **load_kwargs, **extra_kwargs)
        reset_timesteps = False
    else:
        print(f"---- NOUVEAU MODELE {algo_name.upper()} ----")
        model = model_class(
            env=train_env,
            tensorboard_log=logs_dir,
            verbose=1,
            device="auto",
            **algo_cfg,
            **extra_kwargs
        )
        reset_timesteps = True

    # Apprentissage
    try:
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
            tb_log_name=model_name,
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("\nInterruption utilisateur...")
    finally:
        final_path = os.path.join(models_dir, f"{model_name}_final")
        model.save(final_path)
        print(f"Modèle final sauvegardé : {final_path}")
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent on Ice Hockey")
    parser.add_argument("--algo", type=str, default="ppo", help="ppo, partial-ppo, dqn, per-dqn")
    parser.add_argument("--resume", type=str, default=None, help="Chemin vers un .zip")
    args = parser.parse_args()

    train(algo_name=args.algo, resume_path=args.resume)


# Informations entrainement :
#
# rollout/
#   ep_lean_mean : longueur moyenne d'une partie (nb de frames)
#   ep_rew_mean : récompense moyenne obtenu par match
#
# time/
#   fps : vitesse d'entrainement (nb d'image par secondes)
#   iterations : cycle propre au PPO
#   time_elapsed : temps écoulé depuis le début
#   total_timesteps : nombre total d'actions effectuées dans le jeu (par défaut 2048 par env) 
#
# train/
#   approx_kl : divergence KL (mesure à quelle point la nouvelle version de l'agent est différent de l'ancienne)
#   clip_fraction : taux de mise à jour bridées
#   clip_range : taux max de changement autorisé par étape
#   entropy_loss : taux de hasard dans les actions
#   explained_variance : indique si l'agent arrive à prédire le score qu'elle va obtenir (sa compréhension de ses actions et de l'env)
#   value_loss : erreur de l'agent dans sa prédiction des récompenses
