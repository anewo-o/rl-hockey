import gymnasium as gym
import ale_py

env = gym.make("ALE/IceHockey-v5", render_mode="human")
obs, info = env.reset()

done = False
while not done:

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        done = True

env.close()
