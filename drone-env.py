import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class DroneEnv(gym.Env):
    def __init__self(self):
        super(DroneEnv, self).__init__()

        # ACTION SPACE - WHAT THE DRONE CAN DO
        # - use spaces.Box to represent a continuous velocity vector
        # - - (a 2D vector where each component is a real value between -1 and 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)