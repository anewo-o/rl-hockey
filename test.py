import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

env = make_atari_env("ALE/IceHockey-v5", n_envs=1)
env = VecFrameStack(env, n_stack=4)

model = DQN("CnnPolicy", env, verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
)

model.learn(total_timesteps=5000)

print("✅ Training started successfully")

