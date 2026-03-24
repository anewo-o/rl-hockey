# Ice Hockey Atari Game
A Reinforcement Learning (RL) approach...

## Requirements
- `conda install` channels don't have ale-py, must use `pip install`
- `requirements.txt` is not sufficient for Ubuntu (`apt install sld2` etc... wheel hell)
- or use `conda create -n ift702-projet python=3.10` for a more stable Python version
- then for baselines :
    ```bash
    pip install tensor-flow 
    # might break (1.14 last version supported)
    pip uninstall -y setuptools
    pip install setuptools==80.10.2
    pip install -e . --no-build-isolation
    ``` 
- and it did, so you have to use python 3.7 but ale needs 3.9+...
