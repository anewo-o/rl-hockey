import gymnasium as gym
import shimmy
import ale_py

env = gym.make("ALE/IceHockey-v5", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(action)


    if terminated or truncated:
        obs, info = env.reset()

env.close()
