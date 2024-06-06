import gymnasium as gym
from stable_baselines3 import PPO
from drone_env import DroneNavigation

# Register the custom environment
gym.envs.registration.register(
    id='DroneNavigation-v0',
    entry_point='drone_env:DroneNavigation'
)

env = gym.make('DroneNavigation-v0', area_size=10, max_steps=100, save_path='plots_PPO')

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_drone_navigation")

# Load the model
model = PPO.load("ppo_drone_navigation")

# Test the trained model
obs, info = env.reset()
env.render()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, info = env.reset()
        env.render()

env.close()
