import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import os


class DroneNavigation(gym.Env):
    def __init__(self, area_size=10, max_steps=100, num_obstacles=5, save_path='plots'):
        super(DroneNavigation, self).__init__()
        self.area_size = area_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
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
        self.map_plotted = False  # Flag to control the plotting of the probability map

    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < self.num_obstacles:
            pos = (random.randint(0, self.area_size - 1), random.randint(0, self.area_size - 1))
            if pos != tuple(self.drone_pos) and pos != tuple(self.goal_pos):
                obstacles.add(pos)
        return obstacles

    def _generate_probability_map(self):
        prob_map = np.zeros((self.area_size, self.area_size))
        sigma = 1.0  # Standard deviation
        for obs in self.obstacles:
            x0, y0 = obs
            for i in range(self.area_size):
                for j in range(self.area_size):
                    prob_map[j, i] += np.exp(-((i - x0) ** 2 + (j - y0) ** 2) / (2 * sigma ** 2))
        return prob_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([random.randint(0, self.area_size - 1), random.randint(0, self.area_size - 1)],
                                  dtype=np.float32)
        self.goal_pos = np.array([random.randint(0, self.area_size - 1), random.randint(0, self.area_size - 1)],
                                 dtype=np.float32)
        self.steps_taken = 0
        self.obstacles = self._generate_obstacles()
        self.prob_map = self._generate_probability_map()
        self.path = [tuple(self.drone_pos)]
        self.step_count = 0
        self.map_plotted = False  # Reset the flag when the environment is reset
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

        # Check for collisions with obstacles
        if tuple(self.drone_pos) in self.obstacles:
            reward = -50  # Collision with obstacle
            done = True
        elif np.array_equal(self.drone_pos, self.goal_pos):
            reward = 100  # Reached goal
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -10  # Exceeded max steps
            done = True
        else:
            reward = -1  # Penalty for each step taken
            threat_penalty = self.prob_map[int(self.drone_pos[1]), int(self.drone_pos[0])]
            reward -= threat_penalty  # Penalty based on proximity to obstacles
            done = False

        truncated = self.steps_taken >= self.max_steps
        return self.drone_pos, reward, done, truncated, {}

    def render(self, mode='human'):
        if not self.map_plotted:
            self.ax.clear()
            self.ax.set_xlim(0, self.area_size)
            self.ax.set_ylim(0, self.area_size)

            # Plot probability map
            x = np.arange(0, self.area_size)
            y = np.arange(0, self.area_size)
            X, Y = np.meshgrid(x, y)
            c = self.ax.pcolormesh(X, Y, self.prob_map, cmap='coolwarm', shading='auto')

            # Add colorbar only once
            self.colorbar = self.fig.colorbar(c, ax=self.ax)

            # Plot obstacles
            for obs in self.obstacles:
                self.ax.plot(obs[0], obs[1], 'rx', markersize=10)

            # Plot goal
            self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'go', markersize=10)

            # Plot start position
            self.ax.plot(self.path[0][0], self.path[0][1], 'bo', markersize=10)

            self.map_plotted = True

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
