import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Environment setup for the robot
GRID_SIZE = 5
START_POSITION = (0, 0)
GOAL_POSITION = (4, 4)
OBSTACLES = [(1, 1), (2, 2), (3, 3)]
action_size = 4  # up, down, left, right

# Experience replay memory
memory = deque(maxlen=2000)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

def get_reward(state):
    if state == GOAL_POSITION:
        return 10
    elif state in OBSTACLES:
        return -10
    return -1

def get_next_state(state, action):
    x, y = state
    if action == 0 and x > 0:  # up
        x -= 1
    elif action == 1 and x < GRID_SIZE - 1:  # down
        x += 1
    elif action == 2 and y > 0:  # left
        y -= 1
    elif action == 3 and y < GRID_SIZE - 1:  # right
        y += 1
    return (x, y)

def train_dqn(model, episodes, epsilon, epsilon_min, epsilon_decay, gamma, action_size, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Initialize experience replay buffer
    replay_buffer = deque(maxlen=10000)
    
    for episode in range(episodes):
        state = START_POSITION
        state_vector = np.zeros(GRID_SIZE * GRID_SIZE)
        state_vector[state[0] * GRID_SIZE + state[1]] = 1
        total_reward = 0

        model.train()
        for step in range(100):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    action_values = model(state_tensor)
                    action = torch.argmax(action_values).item()

            next_state = get_next_state(state, action)
            next_state_vector = np.zeros(GRID_SIZE * GRID_SIZE)
            next_state_vector[next_state[0] * GRID_SIZE + next_state[1]] = 1
            reward = get_reward(next_state)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.append((state_vector, action, reward, next_state_vector))

            # Only start training when we have enough samples for a batch
            if len(replay_buffer) >= batch_size:
                # Sample batch
                batch = random.sample(replay_buffer, batch_size)
                states = torch.FloatTensor([exp[0] for exp in batch]).to(device)
                actions = torch.LongTensor([exp[1] for exp in batch]).to(device)
                rewards = torch.FloatTensor([exp[2] for exp in batch]).to(device)
                next_states = torch.FloatTensor([exp[3] for exp in batch]).to(device)

                # Compute current Q values
                current_q_values = model(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute next Q values
                with torch.no_grad():
                    next_q_values = model(next_states)
                    max_next_q = next_q_values.max(1)[0]
                    expected_q_values = rewards + gamma * max_next_q

                # Compute loss and update
                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            state_vector = next_state_vector

            if state == GOAL_POSITION or state in OBSTACLES:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
    
    return model

def test_dqn(model):
    model.eval()
    state = START_POSITION
    state_vector = np.zeros(GRID_SIZE * GRID_SIZE)
    state_vector[state[0] * GRID_SIZE + state[1]] = 1
    total_reward = 0
    steps = 0
    path = [state]

    with torch.no_grad():
        while state != GOAL_POSITION and state not in OBSTACLES:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            action_values = model(state_tensor)
            action = torch.argmax(action_values).item()

            next_state = get_next_state(state, action)
            next_state_vector = np.zeros(GRID_SIZE * GRID_SIZE)
            next_state_vector[next_state[0] * GRID_SIZE + next_state[1]] = 1
            reward = get_reward(next_state)
            total_reward += reward
            steps += 1

            state = next_state
            state_vector = next_state_vector
            path.append(state)

            if steps > 100:
                break

    print(f"Test Result: Steps taken = {steps}, Total Reward = {total_reward}")
    print("Path taken:", path)
    return steps, total_reward, path

def visualize_path(path):
    path_x, path_y = zip(*path)
    
    plt.figure(figsize=(8, 8))
    plt.plot(path_y, path_x, 'b-', linewidth=2, label='Path')
    plt.plot(path_y[0], path_x[0], 'go', markersize=15, label='Start')
    plt.plot(path_y[-1], path_x[-1], 'ro', markersize=15, label='Goal')
    
    # Plot obstacles in the environment
    for obs in OBSTACLES:
        plt.plot(obs[1], obs[0], 'ks', markersize=15)
    
    plt.grid(True)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
    plt.legend()
    plt.title('Path Found by DQN Robot')
    plt.show()

# Training parameters
state_size = GRID_SIZE * GRID_SIZE
episodes = 10000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99
batch_size = 32

# Create and train model
model = DQN(state_size, action_size).to(device)
model = train_dqn(model, episodes, epsilon, epsilon_min, epsilon_decay, gamma, action_size, batch_size)

# Test and visualize how the robot performs
steps, total_reward, path = test_dqn(model)
visualize_path(path)