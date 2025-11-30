import os
from stable_baselines3 import PPO
from drone_env import DroneEnv

# set up the paths for the model export
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "ppo_urban_drone")

# instantiate the env
env = DroneEnv()

# define the model (here PPO since it's generally robust for continuous control)
model = PPO("MlpPolicy", env, verbose=1)

# train the agent
print("Training started...")
model.learn(total_timesteps=50000)
print("Training finished!")

# save the model
model.save(model_path)
print(f"SUCCESS: Model saved at {model_path}.zip")