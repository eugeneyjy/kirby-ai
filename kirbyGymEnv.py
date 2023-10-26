import numpy as np
from collections import OrderedDict

from pyboy import PyBoy, WindowEvent
from gymnasium import Env
from gymnasium.spaces import Discrete, Dict, Box

class GameState():
    def __init__(self, pyboy: PyBoy):
        self.game = pyboy.game_wrapper()
        self.kirby_x = pyboy.get_memory_value(0xD05C)
        self.kirby_y = pyboy.get_memory_value(0xD05D)
        self.health = self.game.health
        self.lives_left = self.game.lives_left
        self.score = self.game.score
        self.state = pyboy.get_memory_value(0xD02C)


    def states_dict(self):
        return {
            "kirby_x": self.kirby_x,
            "kirby_y": self.kirby_y,
            "health": self.health,
            "lives_left": self.lives_left,
            "score": self.score,
            "state": self.state
        }


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

        self.button_pressed = [False for _ in range(len(self.actions)//2)]
        self.n_buttons = len(self.button_pressed)

        # Variables to check for truncation condition
        self.step_count = 0
        # self.max_steps = config["max_steps"]

        self.act_freq = config["act_freq"]

        # All posible actions + not doing an action
        self.action_space = Discrete(len(self.actions) + 1)
        # self.observation_space = Dict({
        #     "position": Box(low=np.array([8, 16]), high=np.array([152, 136]), shape=(2,), dtype=np.uint8),
        #     "screen": Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        # })
        self.observation_space = Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)


        head = "headless" if config["headless"] else "SDL2"
        self.pyboy = PyBoy(config["gb_path"], window_type=head, debug=False, game_wrapper=True)
        self.screen = self.pyboy.botsupport_manager().screen()

        self.game = self.pyboy.game_wrapper()
        self.started = False

        # Game Info
        self.curr_game_state = GameState(self.pyboy)
        self.prev_game_state = GameState(self.pyboy)


    def step(self, action: int):
        self.run_emulator_action(action)

        obs = self.get_obs()
        reward = self.get_reward()
        terminated = self.game.game_over()
        # truncated = self.step_count >= self.max_steps
        info = self.game_info()

        self.step_count += 1
        return obs, reward, terminated, False, info


    def reset(self, seed=None, options={}):
        if self.started:
            self.game.reset_game()
        else:
            self.game.start_game()
            self.started = True
        
        for i in range(self.n_buttons):
            # Release pressed button
            if self.button_pressed[i]:
                self.pyboy.send_input(self.actions[i+self.n_buttons])
                self.button_pressed[i] = False

        # Refresh game state      
        self.curr_game_state = GameState(self.pyboy)

        obs = self.get_obs()
        info = self.game_info()
        return obs, info


    def get_reward(self):
        reward = 0
        # Kirby standing still
        if self.curr_game_state.kirby_x == self.prev_game_state.kirby_x:
            reward -= 1
        # Kirby moving right
        elif self.curr_game_state.kirby_x > self.prev_game_state.kirby_x:
            reward += 1
        # Kirby moving right (make screen progress)
        elif self.curr_game_state.kirby_x == 76 and self.button_pressed[3]:
            reward += 5
        # Kirby moving left
        elif self.curr_game_state.kirby_x == 68 and self.button_pressed[2]:
            reward -= 1
        # Kirby moving towards most left
        elif self.curr_game_state.kirby_x < 68:
            reward -= 5


        # Score increased
        if self.curr_game_state.score > self.prev_game_state.score:
            reward += 50

        # Lose health
        if self.curr_game_state.health < self.prev_game_state.health:
            reward -= 100
        # Gain health
        elif self.curr_game_state.health > self.prev_game_state.health and self.curr_game_state.lives_left == self.prev_game_state.lives_left:
            reward += 100

        # Lose live
        if self.curr_game_state.lives_left < self.prev_game_state.lives_left:
            reward -= 1000
        # Gain live
        elif self.curr_game_state.lives_left > self.prev_game_state.lives_left:
            reward += 1000

        # Reached warp star
        if self.curr_game_state.state == 6 and self.prev_game_state.state != 6:
            reward += 1000

        return reward


    # Send input to emulator to perform action accordingly
    def run_emulator_action(self, action: int):
        # Store previous game state (Before action)
        self.prev_game_state = GameState(self.pyboy)

        if action >= 0 and action <= len(self.actions)-1:
            self.pyboy.send_input(self.actions[action])
            # Pressed button
            if action >=0 and action <= (len(self.actions)//2)-1:
                self.button_pressed[action] = True
            # Release button
            else:
                self.button_pressed[action-self.n_buttons] = False
        self.pyboy.tick()

        for _ in range(self.act_freq-1):
            self.pyboy.tick()
        
        # Store current game state (After action)
        self.curr_game_state = GameState(self.pyboy)


    def get_obs(self):
        # obs = OrderedDict([
        #     ("position", np.array([self.curr_game_state.kirby_x, self.curr_game_state.kirby_y])),
        #     ("screen", self.screen.screen_ndarray())
        # ])
        obs = self.screen.screen_ndarray()

        return obs


    def game_info(self):
        info = {
            "gameState": self.curr_game_state.states_dict()
        }

        return info
        
    
    def action_masks(self):
        masks = [1 for _ in range(len(self.actions))]

        for i in range(self.n_buttons):
            if self.button_pressed[i]:
                masks[i] = 0
            else:
                masks[i+self.n_buttons] = 0
        
        # Not doing anything is always possible
        masks.append(1)
        return np.array(masks, dtype=np.int8)