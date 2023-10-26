# Kirby's Dream Land RL Agent
An experiment on training Reinforcement Learning (RL) agent to play Kirby's Dream Land on a GameBoy emulation called [PyBoy](https://github.com/Baekalfen/PyBoy)

# Setup
Download the Kirby's Dream Land ROM and point the `gb_path` in `config.yaml` to that `gb` file.  
```bash
pip install -r requirements.txt
```

# How to train
```bash
python train.py
```

# How to play
Set `agent_enabled=True` in `config.yaml`  
```bash
python play.py
```
