import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

TOTAL_TIMESTEPS = 5000
LEARNING_STARTS = 1000


def main() -> None:
    if LEARNING_STARTS >= TOTAL_TIMESTEPS:
        raise ValueError("LEARNING_STARTS must be lower than TOTAL_TIMESTEPS")

    env = make_atari_env("ALE/IceHockey-v5", n_envs=1)
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=LEARNING_STARTS,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    env.close()

    print("Training completed successfully")


if __name__ == "__main__":
    main()

