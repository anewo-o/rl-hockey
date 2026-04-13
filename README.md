# Ice Hockey Atari Game
A Reinforcement Learning (RL) approach...

## Requirements
- Needs a stable version of Python (3.13) : `conda create -n ift702-projet python=3.13` 
- Incompatibility between OpenAI baselines (Python 3.7) and the Gymnasium (Python 3.9+)
because of tensor-flow 1.14 support only (else pip install -e from baselines repo)

## Dockerize
- Freeze minimal environment with `conda env export --from-history | grep -v "^prefix:" > env.yml`
- Manually add pip package `AutoROM` to dependencies :
    ```yml
    dependencies:
        - pip
        - pip:
            - AutoROM
    ```
- Execute `AutoROM --accept-license --install-dir $CONDA_PREFIX/lib/python3.13/site-packages/ale_py/roms`

## TensorBoard

- **Description :** Visualization toolkit to track different metrics during training (loss, accuracy, etc.).
- **Command to execute :**
  ```bash
  tensorboard --logdir logs/
  ```
- Open your browser : <http://localhost:6006>

## Usage

This project uses unified scripts for training and evaluating multiple Reinforcement Learning algorithms (PPO, Partial-PPO, DQN, PER-DQN). 

### Training (`scripts/train.py`)

The `train.py` script automatically creates `models/`, `logs/`, and `configs/` directories if they do not exist. It loads hyperparameters from `configs/ppo_atari.yml` or `configs/dqn_atari.yml` depending on the algorithm chosen.

**Arguments :**
* `--algo` (required) : The RL algorithm to train. Choices are `ppo`, `partial-ppo`, `dqn`, `per-dqn`.
* `--resume` (optional) : Path to an existing `.zip` model file to resume training.

**Examples :**

1. Train a standard PPO agent from scratch:
    ```bash
    python scripts/train.py --algo ppo
    ```
2. Train a PER-DQN (Prioritized Experience Replay DQN) agent:
    ```bash
    python scripts/train.py --algo per-dqn
    ```
3. Resume an interrupted PPO training from a specific checkpoint:
    ```bash
    python scripts/train.py --algo ppo --resume models/ppo_model_final.zip
    ```

### Evaluation (`scripts/evaluate.py`)
The `evaluate.py` script allows you to test trained models. It supports both visual rendering to watch the agent play, and headless evaluation to compute win/loss statistics.

**Arguments :**
* `--model` (required) : The name of the model file (with or without the .zip extension) located in the `models/` directory, or the direct path.
* `--algo` (required) : The RL algorithm used to train the model. Choices are `ppo`, `partial-ppo`, `dqn`, `per-dqn`.
* `--n-matches` (optional) : Number of matches to play for statistical evaluation. Default is 10.
* `--visual` (optional) : Flag to enable the PyGame/human render mode to watch the match.

**Examples :**

1. Watch your trained PER-DQN agent play 3 matches (Visual Mode):
    ```bash
    python scripts/evaluate.py --model dqn_per_ice_hockey_run1_10000000_steps --algo per-dqn --n-matches 3 --visual
    ```
2. Benchmark a Partial-PPO agent over 50 matches (Headless Mode):
    ```bash
    python scripts/evaluate.py --model ppo_best_model_run1 --algo partial-ppo --n-matches 50
    ```
    