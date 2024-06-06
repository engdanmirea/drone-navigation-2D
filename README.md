# Probabilistic Drone 2D Navigation

This repository contains a custom Gymnasium environment for a drone navigating from a start point to a target, with probabilistic obstacles. The environment is designed to be used with reinforcement learning algorithms, specifically with the Stable-Baselines3 library.

## Features

- **Custom Gymnasium Environment**: Drone navigation with probabilistic obstacles.
- **Dynamic Visualisation**: Real-time 2D visualization of the drone's path, obstacles and probability map.
- **Reinforcement Learning**: Integration with Stable-Baselines3 for training and testing.

## Installation

### Prerequisites

1. **Anaconda**: It's recommended to use Anaconda for managing the Python environment.
2. **Python 3.8**: Ensure you have Python 3.8 installed.

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/probabilistic_drone_nav.git
   cd probabilistic_drone_nav

2. **Create and activate a conda environment**:
    ```bash
   conda create -n drone_navigation python=3.8
   conda activate drone_navigation
   
3. **Install the required packages**:
    ```bash
   pip install -r requirements.txt
   

## Usage

### Train the Agent
To train the agent using the PPO algorithm and test it:

[//]: # (```bash)

[//]: # (python train.py)


## Project Structure


    ```bash
    probabilistic_drone_nav/
    ├── drone_env.py      # Custom environment definition
    ├── train.py                     # Training script
    ├── requirements.txt             # Python dependencies
    ├── README.md                    # Project documentation
    └── plots/                       # Directory to save plots
    
## Environment Details

- **Environment Space**: The collection of possible positions of the drone on a 2D grid.
- **Action Space**:
- **Reward**: +100 for goal reaching, -10 for exceeding max steps, -1 for each step taken, additional penalty
inversely proportional to obstacle proximity.
- **Done**: Episode ends when the drone reaches the goal, hits an obstacle or exceeds max steps. 