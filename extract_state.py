import math

PLAYER_X_IDX = 4
PLAYER_Y_IDX = 5
OPPONENT_X_IDX = 15
OPPONENT_Y_IDX = 16
PUCK_X_IDX = 49
PUCK_Y_IDX = 50
MAX_DIST = 300.0  # safe upper bound
PUCK_POSSESSION_THRESHOLD = 5  # tune as needed
VELOCITY_THRESHOLD = 2


def normalize_state(state):
    normalized = []

    for i, v in enumerate(state):
        if i < 6:  # positions
            normalized.append(v / 255.0)
        elif i < 12:  # velocities
            normalized.append(v / 255.0)
        elif i == len(state) - 1:
            normalized.append(float(v))  # keep 0 or 1
        else:  # distances
            normalized.append(v / MAX_DIST)

    return normalized

def is_valid(x, y):
    return (0 <= x <= 255) and (0 <= y <= 255) and (y != 255)

def get_player(obs):
    x, y = obs[4], obs[5]

    if y == 255:
        return None  # invalid / inactive
    return (x, y)

def compute_velocity(curr, prev):
    return curr[0] - prev[0], curr[1] - prev[1]

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_state(obs, prev_state=None):
    player = (int(obs[PLAYER_X_IDX]), int(obs[PLAYER_Y_IDX]))
    opponent = (int(obs[OPPONENT_X_IDX]), int(obs[OPPONENT_Y_IDX]))
    puck = (int(obs[PUCK_X_IDX]), int(obs[PUCK_Y_IDX]))

    # Handle invalid player
    if not is_valid(*player):
        player = prev_state[0:2] if prev_state else (0, 0)

    # Handle invalid opponent
    if not is_valid(*opponent):
        opponent = prev_state[2:4] if prev_state else (0, 0)

    # Handle invalid puck
    if puck == (0, 0):
        puck = prev_state[4:6] if prev_state else (0, 0)

    return [
        player[0], player[1],
        opponent[0], opponent[1],
        puck[0], puck[1],
    ]

def build_state(raw_state, prev_state=None):

    x_player, y_player, x_opp, y_opp, x_puck, y_puck = raw_state

    player = (x_player, y_player)
    opp = (x_opp, y_opp)
    puck = (x_puck, y_puck)

    if prev_state:
        prev_player = (prev_state[0], prev_state[1])
        prev_opp = (prev_state[2], prev_state[3])
        prev_puck = (prev_state[4], prev_state[5])

        vx_player, vy_player = compute_velocity(player, prev_player)
        vx_opp, vy_opp = compute_velocity(opp, prev_opp)
        vx_puck, vy_puck = compute_velocity(puck, prev_puck)
    else:
        vx_player = vy_player = 0
        vx_opp = vy_opp = 0
        vx_puck = vy_puck = 0

    d_player_opp = distance(player, opp)

    if puck is (0, 0):
        d_player_puck = d_opp_puck = 0
    else:
        d_player_puck = distance(player, puck)
        d_opp_puck = distance(opp, puck)

    if puck is (0, 0):
        has_puck = 0
    else:
        close = d_player_puck < PUCK_POSSESSION_THRESHOLD
        similar_velocity = (
            abs(vx_player - vx_puck) < VELOCITY_THRESHOLD and
            abs(vy_player - vy_puck) < VELOCITY_THRESHOLD
        )
        has_puck = 1 if (close and similar_velocity) else 0

    return [
        # positions (6)
        x_player, y_player,
        x_opp, y_opp,
        x_puck, y_puck,

        # velocities (6)
        vx_player, vy_player,
        vx_opp, vy_opp,
        vx_puck, vy_puck,

        # distances (3)
        d_player_puck,
        d_opp_puck,
        d_player_opp,

        # possession (1)
        has_puck
    ]
