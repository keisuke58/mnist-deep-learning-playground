"""
Step 8: RL Agent that learns to WRITE digits
PPO agent with dense reward, longer training, smarter pen control
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


class DigitWriteEnv:
    """
    Agent draws on 28x28 canvas.
    Actions: 8 directions + pen_down + pen_up + noop (11 actions)
    Dense reward at every step based on local similarity improvement.
    """
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 directions

    def __init__(self, target_images):
        self.targets = target_images
        self.canvas_size = 28
        self.reset()

    def reset(self):
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        self.target = random.choice(self.targets)
        # Start near the top of the digit (find first nonzero row)
        rows = np.where(self.target.sum(axis=1) > 0)[0]
        if len(rows) > 0:
            self.y = int(rows[0]) + 2
            cols = np.where(self.target[self.y] > 0.3)[0]
            self.x = int(cols[len(cols)//2]) if len(cols) > 0 else self.canvas_size // 2
        else:
            self.x = self.canvas_size // 2
            self.y = self.canvas_size // 2
        self.pen_down = True  # Start with pen down
        self.steps = 0
        self.max_steps = 300
        self.prev_mse = np.mean((self.canvas - self.target) ** 2)
        return self._get_state()

    def _get_state(self):
        pos_map = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        pos_map[self.y, self.x] = 1.0
        pen_map = np.full((self.canvas_size, self.canvas_size),
                          float(self.pen_down), dtype=np.float32)
        state = np.stack([self.canvas, self.target, pos_map, pen_map])  # 4 x 28 x 28
        return state

    def step(self, action):
        self.steps += 1

        if action < 8:  # Move in 8 directions
            dy, dx = self.DIRS[action]
            self.y = np.clip(self.y + dy, 0, self.canvas_size - 1)
            self.x = np.clip(self.x + dx, 0, self.canvas_size - 1)
        elif action == 8:  # Pen down
            self.pen_down = True
        elif action == 9:  # Pen up
            self.pen_down = False
        # action == 10: noop

        if self.pen_down:
            # Gaussian-like brush
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = self.y + dy, self.x + dx
                    if 0 <= ny < self.canvas_size and 0 <= nx < self.canvas_size:
                        dist = abs(dy) + abs(dx)
                        if dist == 0:
                            intensity = 0.6
                        elif dist == 1:
                            intensity = 0.3
                        else:
                            intensity = 0.1
                        self.canvas[ny, nx] = min(1.0, self.canvas[ny, nx] + intensity)

        # Dense reward: MSE improvement
        curr_mse = np.mean((self.canvas - self.target) ** 2)
        reward = (self.prev_mse - curr_mse) * 100  # Positive if improving
        self.prev_mse = curr_mse

        # Penalty for drawing where target is blank
        if self.pen_down and self.target[self.y, self.x] < 0.1:
            reward -= 0.05

        # Bonus for drawing where target has ink
        if self.pen_down and self.target[self.y, self.x] > 0.3:
            reward += 0.02

        done = self.steps >= self.max_steps

        if done:
            # Final bonus based on structural similarity
            overlap = np.sum((self.canvas > 0.3) & (self.target > 0.3))
            total_target = max(np.sum(self.target > 0.3), 1)
            coverage = overlap / total_target
            reward += coverage * 5.0

        return self._get_state(), reward, done


class ActorCritic(nn.Module):
    """PPO Actor-Critic network"""
    def __init__(self, n_actions=11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = self(state_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:len(advantages)])]
    return advantages, returns


# Load targets — use digit "7" (easier shape than "1" for brush strokes)
print("=== Step 8: RL Digit Writer (PPO) ===")
mnist = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())

target_digit = 7
targets = []
for img, label in mnist:
    if label == target_digit and len(targets) < 100:
        targets.append(img.squeeze().numpy())
print(f"Target digit: {target_digit} ({len(targets)} examples)")

env = DigitWriteEnv(targets)
model = ActorCritic().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPISODES = 5000
PPO_EPOCHS = 4
CLIP_EPS = 0.2

print(f"Training PPO for {EPISODES} episodes...")
rewards_history = []
best_reward = -float("inf")

for episode in range(EPISODES):
    # Collect trajectory
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    state = env.reset()
    total_reward = 0

    while True:
        action, log_prob, value = model.get_action(state)
        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(float(done))

        state = next_state
        total_reward += reward
        if done:
            break

    rewards_history.append(total_reward)

    # PPO update
    advantages, returns = compute_gae(rewards, values, dones)
    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    old_log_probs_t = torch.FloatTensor(log_probs).to(device)
    advantages_t = torch.FloatTensor(advantages).to(device)
    returns_t = torch.FloatTensor(returns).to(device)

    # Normalize advantages
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        logits, vals = model(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(vals.squeeze(), returns_t)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    if (episode + 1) % 500 == 0:
        avg_r = np.mean(rewards_history[-500:])
        print(f"Episode {episode+1:5d} | Avg Reward: {avg_r:.3f}")

        # Save sample drawings (3 examples)
        imgs = []
        for _ in range(3):
            state = env.reset()
            for _ in range(300):
                action, _, _ = model.get_action(state)
                state, _, done = env.step(action)
                if done:
                    break
            imgs.append(torch.FloatTensor(env.target).unsqueeze(0).unsqueeze(0))
            imgs.append(torch.FloatTensor(env.canvas).unsqueeze(0).unsqueeze(0))
        imgs = torch.cat(imgs)
        save_image(imgs, f"outputs/rl_writer_ep{episode+1:05d}.png", nrow=6, padding=2)
        print(f"  -> Saved: outputs/rl_writer_ep{episode+1:05d}.png")

        if avg_r > best_reward:
            best_reward = avg_r
            torch.save(model.state_dict(), "models/rl_writer.pth")

print(f"\nBest avg reward: {best_reward:.3f}")
print("Saved: models/rl_writer.pth")
