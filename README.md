# 🤖 MazeBot - A Reinforcement Learning Maze Solving Robot

## 📝 Project Overview
This project implements a Reinforcement Learning (RL) based solution for autonomous maze navigation and obstacle avoidance. The initial phase focuses on training a Deep Q-Network (DQN) agent to find optimal paths through a grid-based environment while avoiding obstacles. This is the version 1.0 of the robot that simulates the environment.

## 🎯 Features
- Deep Q-Network (DQN) implementation using PyTorch
- Grid-based environment with customizable size
- Obstacle avoidance capabilities
- Real-time reward visualization
- Path visualization
- GPU acceleration support

## 🛠️ Technical Stack
- Python 3.12
- PyTorch
- NumPy
- Matplotlib
- CUDA (for GPU acceleration)
- Nvidia Jetson Nano
- NVIDIA Jetson Nano Camera IMX219-160

## 🏗️ Project Structure
```
RL-Maze-Solving-Bot/
│
├── Maze-Solving-PytorchV1.py  # Main implementation file
├── Maze-Solving-PytorchV2.py  # Updated implementation with enhancements with ploting Rewars vs Episodes
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12
- CUDA-capable GPU (optional but recommended)
- PyTorch with CUDA support (for GPU acceleration)
- Nvidia Jetson SBCs

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ShakyaDissanayake/RL-Maze-Solving-Bot.git
cd RL-Maze-Solving-Bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage
Run the main script:
```bash
python Maze-Solving-Pytorch.py
```

## 🎮 Environment Details
- Grid Size: 5x5
- Start Position: (0, 0)
- Goal Position: (4, 4)
- Obstacles: Present at positions (1,1), (2,2), and (3,3)

### Reward Structure
- Reaching Goal: +10
- Hitting Obstacle: -10
- Each Step: -1

## 🧠 Model Architecture
The DQN model consists of:
- Input Layer: Grid size squared (25 for 5x5 grid)
- Hidden Layers:
  - Dense layer (128 units) with ReLU activation
  - Dropout layer (0.2)
  - Dense layer (128 units) with ReLU activation
  - Dropout layer (0.2)
  - Dense layer (64 units) with ReLU activation
- Output Layer: 4 units (for up, down, left, right actions)

## 📊 Training Parameters
- Episodes: 500
- Initial Epsilon: 1.0
- Minimum Epsilon: 0.01
- Epsilon Decay: 0.995
- Gamma (Discount Factor): 0.99
- Batch Size: 32
- Learning Rate: 0.001

## 📈 Visualization
The project provides two types of visualizations:
1. Training Progress: Real-time plot of rewards vs episodes
2. Path Visualization: Final path taken by the trained agent

## 🎯 Future Enhancements
- [ ] Implement dynamic maze generation
- [ ] Add support for larger maze environments
- [ ] Integrate with physical robot hardware
- [ ] Add different types of obstacles and rewards
- [ ] Implement more advanced RL algorithms (A3C, PPO)
- [ ] Add real-time visualization of the training process

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- Shakya Dissanayake (@ShakyaDissanayake)

## 🙏 Acknowledgments
- OpenAI for DQN algorithm inspiration
- PyTorch community for excellent documentation

## 📞 Contact
- shakyaanuraj69@gmail.com
- Project Link: [https://github.com/ShakyaDissanayake/RL-Maze-Solving-Bot](https://github.com/ShakyaDissanayake/RL-Maze-Solving-Bot)
