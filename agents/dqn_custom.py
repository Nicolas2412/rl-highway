import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt

from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG

# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class HighwayDQNConfig:
    # Environnement
    env_id: str = "highway-v0"
    seed: int = 42
    # Réseau
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    # Entraînement
    total_timesteps: int = 200_000
    learning_rate: float = 5e-4
    gamma: float = 0.9         # Plus faible que CartPole généralement : horizon court
    batch_size: int = 32
    # Replay buffer
    buffer_capacity: int = 15_000
    learning_starts: int = 200
    train_frequency: int = 1
    # Target network
    target_update_frequency: int = 50
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    # Checkpoints
    checkpoint_dir: str = "/checkpoints"
    checkpoint_frequency: int = 10_000


# ─── Réseau Q pour highway (entrée 2D aplatie) ───────────────────────────────

class HighwayQNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dims):
        super().__init__()
        # obs_shape = (5, 5) → on flatten en 25
        input_dim = int(np.prod(obs_shape))
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x peut être (batch, 5, 5) ou (5, 5) → flatten
        x = x.flatten(start_dim=1) if x.dim() > 2 else x.flatten()
        return self.net(x)


# ─── ReplayBuffer (identique section 5) ──────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, d):
        self.buffer.append((
            np.array(s, dtype=np.float32),
            int(a), float(r),
            np.array(s_, dtype=np.float32),
            float(d)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            np.stack(s), np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32), np.stack(s_),
            np.array(d, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ─── Agent DQN Highway ───────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, cfg: HighwayDQNConfig, obs_shape, n_actions):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_net = HighwayQNetwork(obs_shape, n_actions, cfg.hidden_dims).to(self.device)
        self.target_net = HighwayQNetwork(obs_shape, n_actions, cfg.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.global_step = 0

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    def get_epsilon(self):
        frac = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, obs):
        if random.random() < self.get_epsilon():
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=1).item()
        
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None

        s, a, r, s_, d = self.buffer.sample(self.cfg.batch_size)
        s  = torch.tensor(s,  device=self.device)
        a  = torch.tensor(a,  device=self.device)
        r  = torch.tensor(r,  device=self.device)
        s_ = torch.tensor(s_, device=self.device)
        d  = torch.tensor(d,  device=self.device)

        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(s_).max(dim=1).values
            target_q = r + self.cfg.gamma * max_next_q * (1 - d)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_checkpoint(self, tag="latest"):
        path = os.path.join(self.cfg.checkpoint_dir, f"dqn_highway_{tag}.pt")
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        print(f"Checkpoint chargé depuis {path} (step {self.global_step})")

