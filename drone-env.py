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

        self.state = None
        self.max_steps = 1000
        self.current_step = 0

        def reset(self, seed=None, options=None):
            """
            Function acting as "new game" button for the simulation.
            Purpose:
            1. Initialize the state for a fresh episode
            2. Return the initial observation
            """
            super().reset(seed=seed)

            # RESET INTERNAL STATE
            self.current_step = 0

            # RANDOMIZE DRONE AND TARGET X-Y POSITIONS FOR A NEW EPISODE
            self.drone_pos = np.random.rand(2)
            self.target_pos = np.random.rand(2)

            # ENFORE SPAWNING-RELATED CONSTRAINTS
            # - if accidentally spawns too close to the target, re-roll the target's x-y position
            while np.linalg.norm(self.drone_pos - self.target_pos) < 0.2:
                self.target_pos = np.random.rand(2)
            
            # GENERATE AND RETURN INITIAL OBSERVATION
            observation = self.__get_obs()
            info = {} # empty dictionary for extra debug data (required by API)

            return observation, info