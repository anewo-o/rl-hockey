import torch


def select_minimax_action(q_values):
    """
    q_values: shape (A, B)
    """

    min_q_values, _ = torch.min(q_values, dim=1)

    best_action = torch.argmax(min_q_values).item()

    return best_action