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