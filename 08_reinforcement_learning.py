"""
Step 8: RL Agent that learns to WRITE digits
DQN agent controls a pen on a canvas to draw MNIST-like digits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from collections import deque
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)


# ===== Digit Writing Environment =====
class DigitWriteEnv:
    """
    Agent has a pen on 28x28 canvas.
    Actions: move up/down/left/right, pen up/down
    Reward: similarity to target MNIST digit
    """
    def __init__(self, target_images):
        self.targets = target_images  # list of 28x28 numpy arrays
        self.canvas_size = 28
        self.reset()

    def reset(self):
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        self.x = self.canvas_size // 2
        self.y = self.canvas_size // 2
        self.pen_down = False
        self.target = random.choice(self.targets)
        self.steps = 0
        self.max_steps = 200
        return self._get_state()

    def _get_state(self):
        # State: canvas + target + pen position
        pos_map = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        pos_map[self.y, self.x] = 1.0
        state = np.stack([self.canvas, self.target, pos_map])  # 3 x 28 x 28
        return state

    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=pen_toggle
        self.steps += 1
        if action == 0:
            self.y = max(0, self.y - 1)
        elif action == 1:
            self.y = min(self.canvas_size - 1, self.y + 1)
        elif action == 2:
            self.x = max(0, self.x - 1)
        elif action == 3:
            self.x = min(self.canvas_size - 1, self.x + 1)
        elif action == 4:
            self.pen_down = not self.pen_down

        if self.pen_down:
            # Draw with soft brush
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = self.y + dy, self.x + dx
                    if 0 <= ny < self.canvas_size and 0 <= nx < self.canvas_size:
                        dist = abs(dy) + abs(dx)
                        intensity = 1.0 if dist == 0 else 0.5
                        self.canvas[ny, nx] = min(1.0, self.canvas[ny, nx] + intensity * 0.3)

        done = self.steps >= self.max_steps

        # Reward: negative MSE between canvas and target
        mse = np.mean((self.canvas - self.target) ** 2)
        reward = -mse

        # Bonus for matching pixels
        if done:
            match = np.mean(np.abs(self.canvas - self.target) < 0.3)
            reward += match * 2.0

        return self._get_state(), reward, done


# ===== DQN Agent =====
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 5)  # 5 actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# Load target digits
print("=== Step 8: RL Digit Writer ===")
mnist = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())

# Get clean examples of digit "1" (simplest to draw)
target_digit = 1
targets = []
for img, label in mnist:
    if label == target_digit and len(targets) < 50:
        targets.append(img.squeeze().numpy())
print(f"Target digit: {target_digit} ({len(targets)} examples)")

env = DigitWriteEnv(targets)

# Training
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayBuffer()

EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10

print(f"Training for {EPISODES} episodes...")
rewards_history = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)

    while True:
        # Epsilon-greedy
        if random.random() < eps:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_t).argmax(dim=1).item()

        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        # Train
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones)
            loss = F.smooth_l1_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    rewards_history.append(total_reward)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 100 == 0:
        avg_r = np.mean(rewards_history[-100:])
        print(f"Episode {episode+1:4d} | Avg Reward: {avg_r:.4f} | Eps: {eps:.3f}")

        # Save sample drawing
        state = env.reset()
        for _ in range(200):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_t).argmax(dim=1).item()
            state, _, done = env.step(action)
            if done:
                break
        canvas = torch.FloatTensor(env.canvas).unsqueeze(0).unsqueeze(0)
        target = torch.FloatTensor(env.target).unsqueeze(0).unsqueeze(0)
        save_image(torch.cat([target, canvas], dim=0),
                   f"outputs/rl_writer_ep{episode+1:04d}.png", nrow=2, padding=2)

print("\nSaved outputs/rl_writer_ep*.png (target | agent's drawing)")
torch.save(policy_net.state_dict(), "rl_writer.pth")
