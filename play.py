import yaml
import random

from kirbyGymEnv import KirbyGymEnv
from pyboy import WindowEvent

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

kirbyEnv = KirbyGymEnv(config=config)
obs, info = kirbyEnv.reset()

action = -1
while True:
    if WindowEvent.QUIT in kirbyEnv.pyboy.get_input():
        break
    if config["agent_enabled"]:
        action_masks = kirbyEnv.action_masks()
        action = kirbyEnv.action_space.sample(mask=action_masks)
    obs, reward, terminated, truncated, info = kirbyEnv.step(action)
    print(info)