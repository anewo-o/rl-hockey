import gymnasium as gym
import numpy as np


class IceHockeyShapingWrapper(gym.Wrapper):
    """OCAtari reward shaping shared logic for DQN/PPO pipelines."""

    def __init__(self, env):
        super().__init__(env)
        self.last_min_dist = None
        self.last_puck_target_dist = None
        self.total_base_reward = 0.0
        self.total_shaping_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_min_dist = None
        self.last_puck_target_dist = None
        self.total_base_reward = 0.0
        self.total_shaping_reward = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_reward = reward

        objects = self.env.unwrapped.objects

        my_players = [
            obj for obj in objects
            if getattr(obj, "category", "") == "Player" or type(obj).__name__ == "Player"
        ]
        puck = next(
            (
                obj for obj in objects
                if getattr(obj, "category", "") == "Ball" or type(obj).__name__ == "Ball"
            ),
            None,
        )

        shaping_step_reward = 0.0

        if my_players and puck:
            distances = [np.sqrt((p.x - puck.x) ** 2 + (p.y - puck.y) ** 2) for p in my_players]
            current_min_dist = min(distances)

            if self.last_min_dist is not None:
                if current_min_dist < self.last_min_dist:
                    shaping_step_reward += 0.002
                elif current_min_dist > self.last_min_dist:
                    shaping_step_reward -= 0.002
            self.last_min_dist = current_min_dist

            possession_threshold = 10.0
            if current_min_dist < possession_threshold:
                target_goal_x = 67
                target_goal_y = 155
                current_puck_target_dist = np.sqrt((puck.x - target_goal_x) ** 2 + (puck.y - target_goal_y) ** 2)

                if self.last_puck_target_dist is not None:
                    if current_puck_target_dist < self.last_puck_target_dist:
                        shaping_step_reward += 0.008
                    elif current_puck_target_dist > self.last_puck_target_dist:
                        shaping_step_reward -= 0.008

                self.last_puck_target_dist = current_puck_target_dist
            else:
                self.last_puck_target_dist = None

        custom_reward += shaping_step_reward
        self.total_base_reward += reward
        self.total_shaping_reward += shaping_step_reward

        info = dict(info)
        info["base_reward"] = float(reward)
        info["shaping_reward"] = float(shaping_step_reward)
        info["custom_reward"] = float(custom_reward)

        return obs, custom_reward, terminated, truncated, info
