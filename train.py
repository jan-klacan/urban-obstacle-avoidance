from stable_baselines3 import PPO
from drone_env import DroneEnv

# instantiate the env
env = DroneEnv()

# define the model (here PPO since it's generally robust for continuous control)
model = PPO("MlpPolicy", env, verbose=1)

# train the agent
print("Training started...")
model.learn(total_timesteps=50000)
print("Training finished!")

# save the model
model.save("ppo_urban_drone")