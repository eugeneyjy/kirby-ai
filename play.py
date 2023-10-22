import yaml
import random

from kirbyGymEnv import KirbyGymEnv
from pyboy import WindowEvent

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

kirbyEnv = KirbyGymEnv(config=config)
kirbyEnv.game.start_game()

action = -1
while True:
    if WindowEvent.QUIT in kirbyEnv.pyboy.get_input():
        break
    if config["agent_enabled"]:
        action = random.randint(0, len(kirbyEnv.actions)-1)
    kirbyEnv.step(action)