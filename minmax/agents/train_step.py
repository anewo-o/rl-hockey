import torch
import torch.nn.functional as F


def train_step(q_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions_a, actions_b, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions_a = torch.tensor(actions_a, dtype=torch.long).to(device)
    actions_b = torch.tensor(actions_b, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = q_net(states)

    batch_indices = torch.arange(batch_size).to(device)
    current_q = q_values[batch_indices, actions_a, actions_b]

    # Q(s', a', b')
    with torch.no_grad():
        next_q_values = target_net(next_states)

        # min over opponent actions b'
        min_over_b, _ = torch.min(next_q_values, dim=2)

        # max over agent's actions a'
        max_min_q, _ = torch.max(min_over_b, dim=1)

        targets = rewards + gamma * max_min_q * (1 - dones)

    loss = F.mse_loss(current_q, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()