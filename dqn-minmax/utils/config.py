# Device
DEVICE = "cuda"  # or "cpu"

# Training
NUM_EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-3

# Replay buffer
BUFFER_CAPACITY = 10000

# Epsilon (exploration)
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

# Target network
TARGET_UPDATE_FREQ = 1000  # steps

# Model
STATE_DIM = 128
NUM_ACTIONS = 18

# Saving
SAVE_FREQ = 50
MODEL_PATH = "models/"