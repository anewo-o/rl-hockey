import ale_py
import os
import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

path_to_model = os.path.join("models", "ppo_ice_hockey_v1")

def test_model():
    # Environnement de test 
    print("Chargement de l'environnement...")
    env = make_atari_env("ALE/IceHockey-v5", n_envs=1, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    # Chargement du modèle 
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
    test_model()