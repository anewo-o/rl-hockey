import ale_py
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def train():

    models_dir = "models"
    model_name = "ppo_ice_hockey_v1"
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 4 environnements en parallèle (accélére l'entraînement)
    env = make_atari_env("ALE/IceHockey-v5", n_envs=4, seed=0)

    # 4 frames emplilées pour que l'agent comprenne le mouvement
    env = VecFrameStack(env, n_stack=4)

    # Modèle PPO avec CNN
    model = PPO("CnnPolicy", env, verbose=1, device="auto")

    # 4. Entraînement
    print("Début de l'entraînement...")
    try:
        model.learn(total_timesteps=10000)
    except KeyboardInterrupt:
        print("\nInterruption, sauvegarde du modèle...")
    finally:
        # 5. Sauvegarde de modèle
        model.save(os.path.join(models_dir, model_name))
        print("Modèle sauvegardé")
        env.close()

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
