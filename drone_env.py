import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import os


class DroneNavigation(gym.Env):
    def __init__(self, area_size=10, max_steps=100, save_path='plots'):
        super(DroneNavigation, self).__init__()
        self.area_size = area_size
        self.max_steps = max_steps
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.area_size - 1, shape=(2,), dtype=np.float32)

        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")

        # Initialize environment
        self.reset()

        # Initialize plotting
        self.fig, self.ax = plt.subplots()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([random.randint(0, self.area_size - 1), random.randint(0, self.area_size - 1)],
                                  dtype=np.float32)
        self.goal_pos = np.array([random.randint(0, self.area_size - 1), random.randint(0, self.area_size - 1)],
                                 dtype=np.float32)
        self.steps_taken = 0
        self.path = [tuple(self.drone_pos)]
        self.step_count = 0
        return self.drone_pos, {}

    def step(self, action):
        if action == 0:  # up
            self.drone_pos[1] = min(self.drone_pos[1] + 1, self.area_size - 1)
        elif action == 1:  # down
            self.drone_pos[1] = max(self.drone_pos[1] - 1, 0)
        elif action == 2:  # left
            self.drone_pos[0] = max(self.drone_pos[0] - 1, 0)
        elif action == 3:  # right
            self.drone_pos[0] = min(self.drone_pos[0] + 1, self.area_size - 1)

        self.steps_taken += 1
        self.path.append(tuple(self.drone_pos))
        self.step_count += 1

        # Calculate reward
        if np.array_equal(self.drone_pos, self.goal_pos):
            reward = 100  # Reached goal
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -10  # Exceeded max steps
            done = True
        else:
            reward = -1  # Penalize for each step taken
            done = False

        truncated = self.steps_taken >= self.max_steps
        return self.drone_pos, reward, done, truncated, {}

    def render(self, mode='human'):
        self.ax.clear()
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)

        # Plot goal
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'go', markersize=10)

        # Plot start position
        self.ax.plot(self.path[0][0], self.path[0][1], 'bo', markersize=10)

        # Plot path
        path_np = np.array(self.path)
        self.ax.plot(path_np[:, 0], path_np[:, 1], 'b-', linewidth=2)

        # Plot current drone position
        self.ax.plot(self.drone_pos[0], self.drone_pos[1], 'bo')

        plt.title(f'Step: {self.step_count}')
        plt.pause(0.001)

        # Save plot
        plt.savefig(os.path.join(self.save_path, f'step_{self.step_count:04d}.png'))

    def close(self):
        plt.close(self.fig)
