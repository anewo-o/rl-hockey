# Setup

python3 -m venv rl-marl-env
source rl-marl-env/bin/activate

pip install -r requirements.txt

# Install ROMs

pip install autorom
AutoROM --accept-license

# Set ROM path

export ALE_PY_ROM_DIR=.../multi_agent_ale_py/roms

# Run

python index.py
