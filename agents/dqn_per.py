"""
Simple DQN + Prioritized Experience Replay.

This file intentionally mirrors the existing custom DQN structure so the PER
variant stays isolated from the vanilla implementation.
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent


@dataclass
class HighwayPERConfig:
    env_id: str = "highway-v0"
    seed: int = 42
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    total_timesteps: int = 200_000
    learning_rate: float = 5e-4
    gamma: float = 0.9
    batch_size: int = 32
    buffer_capacity: int = 15_000
    learning_starts: int = 200
    train_frequency: int = 1
    target_update_frequency: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    double_dqn: bool = False
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 10_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-5


class HighwayQNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dims):
        super().__init__()
        input_dim = int(np.prod(obs_shape))
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1) if x.dim() > 1 else x.flatten())


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, priority_epsilon: float):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, terminated):
        max_priority = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        self.buffer[self.position] = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(terminated),
        )
        self.priorities[self.position] = max(max_priority, self.priority_epsilon)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        current_priorities = self.priorities[: self.size]
        scaled_priorities = np.power(current_priorities, self.alpha, dtype=np.float32)
        prob_sum = float(np.sum(scaled_priorities))
        if prob_sum <= 0.0:
            probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float32)
        else:
            probabilities = scaled_priorities / prob_sum

        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        weights = np.power(self.size * probabilities[indices], -beta, dtype=np.float32)
        weights /= weights.max()

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        updated = np.abs(td_errors) + self.priority_epsilon
        self.priorities[np.asarray(indices, dtype=np.int64)] = updated.astype(np.float32)

    def __len__(self):
        return self.size


class PERDQNAgent(BaseAgent):
    def __init__(self, cfg: HighwayPERConfig, obs_shape, n_actions):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = HighwayQNetwork(obs_shape, n_actions, cfg.hidden_dims).to(self.device)
        self.target_net = HighwayQNetwork(obs_shape, n_actions, cfg.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.buffer = PrioritizedReplayBuffer(
            capacity=cfg.buffer_capacity,
            alpha=cfg.per_alpha,
            priority_epsilon=cfg.per_epsilon,
        )
        self.global_step = 0

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    def act(self, obs, epsilon=None) -> int:
        return self._greedy(obs)

    def update(self, obs=None, action=None, reward=None, terminated=None, next_obs=None) -> Optional[float]:
        if len(self.buffer) < self.cfg.batch_size:
            return None

        beta = self.get_beta()
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(
            self.cfg.batch_size, beta=beta
        )

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            else:
                max_next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.cfg.gamma * max_next_q * (1.0 - dones_t)

        td_errors = target_q - current_q
        loss = (weights_t * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        return float(loss.item())

    def train(self, env, total_timesteps=10_000, seed=None, log_dir=None, run_name=None):
        raise NotImplementedError("Use core_task/train_dqn_per.py for vectorized PER training.")

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path):
        self.load_checkpoint(path)

    @property
    def needs_training(self):
        return True

    def get_epsilon(self) -> float:
        fraction = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def get_beta(self) -> float:
        fraction = min(1.0, self.global_step / self.cfg.total_timesteps)
        return self.cfg.per_beta_start + fraction * (self.cfg.per_beta_end - self.cfg.per_beta_start)

    def select_action(self, obs) -> int:
        if random.random() < self.get_epsilon():
            return random.randint(0, self.n_actions - 1)
        return self._greedy(obs)

    def select_actions_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        batch_size = len(obs_batch)
        actions = np.array([random.randint(0, self.n_actions - 1) for _ in range(batch_size)])
        greedy_mask = np.random.rand(batch_size) >= self.get_epsilon()
        if greedy_mask.any():
            obs_t = torch.tensor(obs_batch[greedy_mask], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions[greedy_mask] = self.q_net(obs_t).argmax(dim=1).cpu().numpy()
        return actions

    def _greedy(self, obs) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() < 2:
            obs_t = obs_t.unsqueeze(0)
        elif obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=1).item()

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_checkpoint(self, tag: str = "latest") -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.cfg.checkpoint_dir, f"{timestamp}_per_dqn_{tag}.pt")
        self.save(path)
        return path

    def load_checkpoint(self, path: str, show:bool=True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        if show:
            print(f"Checkpoint charge depuis {path} (step {self.global_step})")
