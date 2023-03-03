# how to define a custom gym environment
'''
import gym
from gym import spaces

class CustomEnv(gym.Env):
  def __init__(self, arg1, arg2, ...):

    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...
'''
import gym
from gym import spaces
import numpy as np


class pm_env(gym.Env):
    def __init__(self, args):
        self.stock_dim = stock_dim
        self.techlist_dim = techlist_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.techlist_dim*self.stock_dim,), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1+self.stock_dim,), dtype=np.float64)

    def step(self, action):
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        return e, obs
