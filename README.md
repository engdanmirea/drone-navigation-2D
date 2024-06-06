# Probabilistic Drone 2D Navigation

This repository contains a custom Gymnasium environment for a drone navigating from a start point to a target, with probabilistic obstacles. The environment is designed to be used with reinforcement learning algorithms, specifically with the Stable-Baselines3 library.

## Features

- **Custom Gymnasium Environment**: Drone navigation with probabilistic obstacles.
- **Dynamic Visualisation**: Real-time 2D visualization of the drone's path, obstacles and probability map.
- **Reinforcement Learning**: Integration with Stable-Baselines3 for training and testing.

## Installation

### Prerequisites

1. **Anaconda**: It's recommended to use Anaconda for managing the Python environment.
2. **Python 3.11**: Ensure you have Python 3.11 installed.

### Steps

1. **Clone the repository**:
   ```bash
   https://github.com/engdanmirea/drone-navigation-2D.git
   cd drone-navigation-2D

2. **Create and activate a conda environment**:
    ```bash
   conda create -n <env_name> python=3.11.9
   conda activate <env_name>
   
3. **Install the required packages**:
    ```bash
   pip install -r requirements.txt
   

## Usage

### Train the Agent
Train the agent using the PPO or DQN algorithm and test it in a generated scenario:

 '```bash
python train_ppo.py
python train_dqn.py


## Project Structure


    ```bash
    probabilistic_drone_nav/
    ├── drone_env.py      # Custom environment definition
    ├── train_ppo.py                 # Training the agent with PPO
    ├── train_dqn.py                 # Training the agent with DQN
    ├── requirements.txt             # Python dependencies
    ├── README.md                    # Project documentation
    └── plots/                       # Directory to save plots
    
## Environment Details

- **Environment Space**: The collection of possible positions of the drone on a 2D grid.
- **Action Space**: Discrete actions for moving up, down, left and right.
- **Reward**: +100 for goal reaching, -10 for exceeding max steps, -1 for each step taken, -50 for obstacle collision, additional penalty
 for proximity to obstacles based on the probabilistic threat map.
- **Done**: Episode ends when the drone reaches the goal, hits an obstacle or exceeds max steps. 

## Algortihms
Both PPO and DQN algorithms from Stable-Baselines3 are employed for training the agent.
