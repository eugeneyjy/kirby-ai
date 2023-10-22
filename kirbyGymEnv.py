from pyboy import PyBoy, WindowEvent
from gymnasium import Env

class KirbyGymEnv(Env):
    def __init__(self, config=None):
        self.actions = [
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.pyboy = PyBoy(config["gb_path"], debug=False, game_wrapper=True)
        self.game = self.pyboy.game_wrapper()

    def step(self, action):
        self.run_emulator_action(action)


    def run_emulator_action(self, action):
        if action >= 0 and action <= len(self.actions)-1:
            self.pyboy.send_input(self.actions[action])
        self.pyboy.tick()