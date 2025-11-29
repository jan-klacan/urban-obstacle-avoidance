import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        # ACTION SPACE - WHAT THE DRONE CAN DO
        # - use spaces.Box to represent a continous velocity vector (2D)
        # - each component a real value between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # OBSERVATION SPACE - WHAT THE DRONE CAN SEE
        # - use spaces.Box to represent a 12D vector
        # - - drone position: 2 numbers
        # - - target position: 2 numbers
        # - - LIDAR readings: 8 numbers
        # - all values are normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape= (12,),
            dtype=np.float32
        )