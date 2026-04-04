import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from ocatari.core import OCAtari

from src.wrappers import IceHockeyShapingWrapper

def create_ice_hockey_env(n_envs=1, render_mode=None, n_stack=4, seed=42):
    def make_env(rank):
        def _init():
            # 1. OCAtari : On force obs_mode="ori" pour garantir la sortie des pixels RGB
            env = OCAtari("ALE/IceHockey-v5", mode="ram", hud=True, obs_mode="ori", render_mode=render_mode)
            
            # Patch OCAtari / SB3
            if not hasattr(env.unwrapped, "ale"):
                env.unwrapped.ale = env.unwrapped._env.unwrapped.ale
            if not hasattr(env.unwrapped, "get_action_meanings"):
                env.unwrapped.get_action_meanings = env.unwrapped._env.unwrapped.get_action_meanings

            # Wrapper pour le reward shaping 
            env = IceHockeyShapingWrapper(env)

            # Pour avoir les stats rollout/
            env = Monitor(env)
            
            # Pour l'image donnée au PPO
            env = AtariWrapper(env, clip_reward=False, terminal_on_life_loss=False)
            
            # Patch pour forcer le format gymnasium officiel
            env.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
            env.action_space = gym.spaces.Discrete(env.action_space.n)
            
            return env
        return _init

    # Création des fonctions d'initialisation pour chaque processus
    env_fns = [make_env(i) for i in range(n_envs)]

    # Vectorisation
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Empilement des frames pour la perception du mouvement
    env = VecFrameStack(env, n_stack=n_stack)
    
    # Transposition pour PyTorch (Canaux, Hauteur, Largeur)
    env = VecTransposeImage(env)

    env.seed(seed)

    return env