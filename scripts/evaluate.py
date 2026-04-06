import os
import sys
import time
import ale_py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from src.env_utils import create_ice_hockey_env

# Choix du modèle à tester
MODEL = "best_model_run3"
#MODEL = "checkpoints/ppo_ice_hockey_run3_15000000_steps"

def evaluate_model(model_name):
    path_to_model = os.path.join("models", model_name)

    # Environnement de test 
    print("Chargement de l'environnement...")
    env = create_ice_hockey_env(n_envs=1, render_mode="human", seed=42)

    print("Chargement du modèle PPO...")
    try:
        model = PPO.load(path_to_model)
    except FileNotFoundError:
        print("Erreur : Le modèle est introuvable")
        return

    obs = env.reset()

    try:
        while True: 
            # L'IA prédit la meilleure action à partir de l'observation
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            # Vitesse de jeu
            time.sleep(0.01) 

            if dones:
                print("Fin du match, réinitialisation...")
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nTest arrêté par l'utilisateur.")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate_model(MODEL)