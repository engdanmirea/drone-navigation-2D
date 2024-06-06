import gymnasium as gym
from stable_baselines3 import DQN
from drone_env import DroneNavigation

# Register the custom environment
gym.envs.registration.register(
    id='DroneNavigation-v0',
    entry_point='drone_env:DroneNavigation'
)

env = gym.make('DroneNavigation-v0', area_size=10, max_steps=100, num_obstacles=5, save_path='plots_DQN')

# Create the DQN model
model = DQN('MlpPolicy', env, verbose=1, buffer_size=50000, learning_rate=1e-3, batch_size=32, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.02)

# Train the model
model.learn(total_timesteps=50000)

# Save the model
model.save("dqn_drone_navigation")

# Load the model
model = DQN.load("dqn_drone_navigation")

# Test the trained model
obs, info = env.reset()
env.render()
for _ in range(200):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, info = env.reset()
        env.render()

env.close()