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
        #   use spaces.Box to represent a 12D vector
        #       drone position: 2 numbers
        #       target position: 2 numbers
        #       LIDAR readings: 8 numbers
        #   all values are normalized to [0, 1]
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
        observation = self._get_obs()
        info = {} # empty dictionary for extra debug data (required by API)

        return observation, info
    

    def _get_obs(self):
        """
        The "camera" of the environment.
        Purpose: convert the internal state variables into the 12D observation vector.
        """
        
        # Default: the drone sees no obstacle within its max range (1.0) in any of the 8 directions
        lidar_readings = np.full(8, 1.0)

        # BUILD THE FIRST OBSERVATION VECTOR
        obs = np.concatenate((self.drone_pos, self.target_pos, lidar_readings))

        return obs.astype(np.float32)
    

    def step(self, action):
        """
        Advance the environment by one time step given a drone action.
        Returns:
            tuple:
                obs (np.ndarray): 12D normalized observation vector after the action.
                reward (float): Scalar reward based on distance to the target.
                terminated (bool): True if the goal was reached this step.
                truncated (bool): True if the episode ended due to time limit.
                info (dict): Extra diagnostic information (unused here).
        """

        # UPDATE TIME
        self.current_step += 1

        # APPLY ACTION (VELOCITY) TO POSITION
        # (Scale action by 0.05 so it doesn't teleport)
        self.drone_pos += action * 0.05
        self.drone_pos = np.clip(self.drone_pos, 0, 1)

        # CALCULATE DISTANCE TO TARGET
        dist_to_target = np.linalg.norm(self.drone_pos - self.target_pos)
        
        # CALCULATE REWARD
        reward = -dist_to_target
        terminated = False

        # CHECK TERMINATION CONDITION (SUCCESS)
        if dist_to_target < 0.05:
            reward += 100
            terminated = True

        # CHECK TIME LIMIT
        truncated = self.current_step >= self.max_steps

        # BUILD NEW OBSERVATION
        obs = self._get_obs()
        info = {} # empty dictionary for extra debug data (required by API)

        return obs, reward, terminated, truncated, info
        

    def render(self):
        """
        Visualization function.
        """

        plt.scatter(self.drone_pos[0], self.drone_pos[1], c="blue", label="Drone")
        plt.scatter(self.target_pos[0], self.target_pos[1], c="green", label="Target")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.pause(0.01)
        plt.clf()