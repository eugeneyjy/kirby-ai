import yaml

from kirbyGymEnv import KirbyGymEnv

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank, env_config, seed=42):
    def _init():
        env = KirbyGymEnv(env_config)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.full_load(file)

    vec_env = DummyVecEnv([make_env(i, config) for i in range(config["n_env"])])
    model = MaskablePPO('CnnPolicy', vec_env, verbose=1, n_steps=config["n_steps"], tensorboard_log="./ppo_kirby_tensorboard/")
    model.learn(total_timesteps=config["n_env"]*config["n_steps"]*config["n_episodes"])
    print("Finished Training")