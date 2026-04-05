import os
import random
import cv2
import torch
import torch.optim as optim

from pettingzoo.atari import ice_hockey_v2

from agents.replay_buffer import MinimaxReplayBuffer
from agents.minimax_q_network import MinimaxQNetwork
from agents.action_selection import select_minimax_action
from agents.train_step import train_step
from utils import config
from utils.logger import TrainingLogger


os.makedirs(config.MODEL_PATH, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
)

q_net = MinimaxQNetwork(
    state_dim=config.STATE_DIM,
    num_actions=config.NUM_ACTIONS
).to(device)

q_net.train()

target_net = MinimaxQNetwork(
    state_dim=config.STATE_DIM,
    num_actions=config.NUM_ACTIONS
).to(device)

target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=config.LEARNING_RATE)

buffer = MinimaxReplayBuffer(capacity=config.BUFFER_CAPACITY)

batch_size = config.BATCH_SIZE
gamma = config.GAMMA

epsilon = config.EPSILON_START
epsilon_min = config.EPSILON_MIN
epsilon_decay = config.EPSILON_DECAY

num_episodes = config.NUM_EPISODES


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, config.NUM_ACTIONS - 1)

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_net(state)[0]

    return select_minimax_action(q_values)


def choose_opponent_action(env, agent):
    return env.action_space(agent).sample()

def main():

    print(f"Using device: {device}")

    env = ice_hockey_v2.env(obs_type="ram")
    step_count = 0
    global epsilon
    logger = TrainingLogger()


    for episode in range(num_episodes):
        env.reset()

        last_state = None
        last_action_a = None
        reward_A = 0
        reward_B = 0
        episode_length = 0


        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                action = None

            elif agent == "first_0":
                last_state = obs
                last_action_a = choose_action(obs, epsilon)
                action = last_action_a
                reward_A += reward

            else:
                action_b = choose_opponent_action(env, agent)
                action = action_b
                reward_B += reward

                next_state = obs

                if last_state is not None and last_action_a is not None:
                    buffer.push(
                        last_state,
                        last_action_a,
                        action_b,
                        reward,
                        next_state,
                        done
                    )
                    loss = train_step(q_net, target_net, optimizer, buffer, batch_size, gamma, device)
                    logger.log_loss(loss)

                episode_length += 1

                step_count += 1
                logger.log_step()

                if step_count > 0 and step_count % config.TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(q_net.state_dict())
                
            env.step(action)
            
        logger.log_episode(reward_A, reward_B, episode_length)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 2 == 0:
            logger.print(epsilon)

        if episode % config.SAVE_FREQ == 0:
            torch.save(q_net.state_dict(), f"{config.MODEL_PATH}/model_ep{episode}.pth")
            
    env.close()

if __name__ == "__main__":
    main()