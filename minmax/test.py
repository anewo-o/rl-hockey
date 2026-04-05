import os
from pettingzoo.atari import ice_hockey_v2
import random
import cv2


def render_frame(env):
    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Ice Hockey", frame)
    cv2.waitKey(1)

def choose_action():
    return random.randint(0, 17)

def choose_opponent_action(env, agent):
    return env.action_space(agent).sample()

def main():

    env = ice_hockey_v2.env(render_mode="rgb_array", obs_type="ram")
    env.reset()


    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None

        elif agent == "first_0":
            action = choose_action()

        elif agent == "second_0":
            action = choose_opponent_action(env, agent)

        render_frame(env)

        print(f"Agent: {agent}, Action: {action}, Reward: {reward}")

        env.step(action)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()