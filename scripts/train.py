import os
import sys
import yaml
import ale_py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from src.env_utils import create_ice_hockey_env
from typing import Callable


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


def load_config(config_path):
    """Charge le fichier YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train():

    models_dir = os.path.join("models")
    logs_dir = os.path.join("logs")
    config_path = os.path.join("configs", "ppo_atari.yml")
    
    # Créer les dossiers s'ils n'existent pas
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Chargement des hyperparamètres
    print(f"Chargement de la configuration depuis {config_path}...")
    config = load_config(config_path)
    env_cfg = config["environment"]
    ppo_cfg = config["ppo"]
    train_cfg = config["training"]
    print(f"Learning rate de départ de {ppo_cfg['learning_rate']})")
    ppo_cfg["learning_rate"] = linear_schedule(ppo_cfg["learning_rate"])

    #model_name = "ppo_ice_hockey_v1"

    print("Initialisation des environnements d'entraînement...")
    train_env = create_ice_hockey_env(n_envs=env_cfg["n_envs"], render_mode=None, seed=0)

    # Environnement pour tester le modèle et déterminer le meilleur à sauvegarder
    print("Initialisation de l'environnement d'évaluation...")
    eval_env = create_ice_hockey_env(n_envs=1, render_mode=None, seed=42)

    # Config du callback pour gérer l'évaluation périodique du modèle
    # gère automatiquement le passage du modèle entre train_env et eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=max(1, train_cfg["eval_freq"] // env_cfg["n_envs"]), # Ajustement par rapport au nb d'env
        n_eval_episodes=5, # nombre de partie jouer pour connaitre plus précisement la performance
        deterministic=True,
        render=False
    )

    print("Création du modèle PPO...")
    model = PPO(
        env=train_env, 
        verbose=1, 
        device="auto",
        tensorboard_log=logs_dir,
        **ppo_cfg
    )

    print(f"Début de l'entraînement pour {train_cfg['total_timesteps']} timesteps...")
    try:
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=eval_callback,
            tb_log_name="PPO_IceHockey_run1"
        )
    except KeyboardInterrupt:
        print("\nInterruption, sauvegarde du modèle...")
    finally:
        # Sauvegarde de modèle
        final_model_path = os.path.join(models_dir, "ppo_ice_hockey_final")
        model.save(final_model_path)
        print(f"Modèle final sauvegardé sous {final_model_path}")
        print(f"Le MEILLEUR modèle a été sauvegardé automatiquement sous {models_dir}/best_model.zip")
        
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    train()

# Informations entrainement :
#
# rollout/
#   ep_lean_mean : longueur moyenne d'un partie (nb de frames)
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
