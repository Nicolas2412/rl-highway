"""
Defines the DQNAgent class implementing a custom DQN
Linear Epsilon Decay
Includes a flag to activate double DQN
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional
from agents.base_agent import BaseAgent
import gymnasium as gym 
import highway_env  # noqa: F401
from shared_core_config import SHARED_CORE_CONFIG

# Config class, is also used by the sb3 version
@dataclass
class HighwayDQNConfig:
    env_id: str = "highway-v0"
    seed: int = 42
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    total_timesteps: int = 200_000
    learning_rate: float = 0.0001432249371823026
    gamma: float = 0.8296389588638785
    batch_size: int = 64
    buffer_capacity: int = 30000
    learning_starts: int = 200
    train_frequency: int = 1
    target_update_frequency: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 150_000
    double_dqn: bool = False
    checkpoint_dir: str = "../rl-highway/checkpoints"
    checkpoint_frequency: int = 10_000

# Q Network class
class HighwayQNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dims):
        super().__init__()
        input_dim = int(np.prod(obs_shape))
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flattens every dimension except the batch
        return self.net(x.flatten(start_dim=1) if x.dim() > 1 else x.flatten())


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, terminated):
        self.buffer.append((
            np.array(s, dtype=np.float32),
            int(a),
            float(r),
            np.array(s_, dtype=np.float32),
            float(terminated),   # terminated only, not done (for bootstrap)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s_),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Agent
class DQNAgent(BaseAgent):
    def __init__(self, cfg: HighwayDQNConfig, obs_shape, n_actions):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = HighwayQNetwork(obs_shape, n_actions, cfg.hidden_dims).to(self.device)
        self.target_net = HighwayQNetwork(obs_shape, n_actions, cfg.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.buffer    = ReplayBuffer(cfg.buffer_capacity)
        self.global_step = 0

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # BaseAgent interface

    def act(self, obs, epsilon=None) -> int:
        """Greedy selection (for evaluation), ignoring epsilon"""
        return self._greedy(obs)

    def update(self, obs=None, action=None, reward=None,
            terminated=None, next_obs=None) -> Optional[float]:
        """
        Samples a mini-batch from the buffer and performs a network update.
        Positional arguments are retained to match the BaseAgent signature; 
        pushing to the buffer is handled upstream by the training loop 
        (train_highway_dqn / train_vectorized / train).
        Returns the loss, or None if the buffer is too small.
        """
        if len(self.buffer) < self.cfg.batch_size:
            return None

        s, a, r, s_, d = self.buffer.sample(self.cfg.batch_size)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.long,    device=self.device)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device)
        s_ = torch.tensor(s_, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device)

        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                best_a     = self.q_net(s_).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(s_).gather(1, best_a).squeeze(1)
            else:
                max_next_q = self.target_net(s_).max(dim=1).values
            target_q = r + self.cfg.gamma * max_next_q * (1.0 - d)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def train(self, env, total_timesteps: int = 10_000,
            seed: Optional[int] = None,
            log_dir: Optional[str] = None,
            run_name: Optional[str] = None) -> None:
        """
        Episodic training loop (BaseAgent interface).
        Used by evaluate_over_seeds; operates on a single (non-vectorized) env, 
        unlike train_vectorized.

        Hyperparameters (lr, gamma, epsilon...) are read from self.cfg.
        TensorBoard: logs to log_dir/run_name if provided.
        """

        # Reproductibility
        _seed = seed if seed is not None else self.cfg.seed
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)

        # TensorBoard
        writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)

        # Environment configuration
        # The env is given via evaluate_over_seeds ; we configure it here if
        # it exposes configure().
        if hasattr(env.unwrapped, "configure"):
            env.unwrapped.configure(SHARED_CORE_CONFIG)

        obs, _ = env.reset(seed=_seed)
        step = self.global_step
        
        episode_rewards = []
        losses = []
        episode_count = 0
        collision_count = 0
        
        ep_reward = 0.0
        ep_len = 0
        ep_speeds = []
        ep_sub_rewards = {}
        
        initial_step = step
        start_time = time.time()
        while step < total_timesteps:
            
            action = self.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            
            if "speed" in info:
                ep_speeds.append(info["speed"])
            if "rewards" in info:
                for key, val in info["rewards"].items():
                    if key not in ep_sub_rewards:
                        ep_sub_rewards[key] = []
                    ep_sub_rewards[key].append(val)
                    
            self.buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward
            ep_len += 1
            step += 1
            self.global_step = step
            
            # Update Network
            if step >= self.cfg.learning_starts and step % self.cfg.train_frequency == 0:
                loss = self.update()
                if loss is not None:
                    losses.append(loss)
                    if writer is not None:
                        writer.add_scalar("train/loss", loss, step)
            
            # Synchronyze target network
            if step % self.cfg.target_update_frequency == 0:
                self.sync_target_network()

            # Checkpoint save
            if self.cfg.checkpoint_frequency > 0 and step % self.cfg.checkpoint_frequency == 0:
                self.save_checkpoint(tag=f"step{step}")
                    
            if done or truncated:
                
                episode_rewards.append(ep_reward)

                # Logging TensorBoard by episode
                episode_count += 1
                if info.get("crashed", False):
                    collision_count += 1
                collision_rate = collision_count / episode_count
                
                elapsed_time = time.time() - start_time
                fps = int((step - initial_step) / max(elapsed_time, 1e-6))
                
                if writer is not None:
                    writer.add_scalar("rollout/ep_rew_mean", ep_reward, step)
                    writer.add_scalar("rollout/ep_len_mean", ep_len, step)
                    writer.add_scalar("rollout/exploration_rate",self.get_epsilon(), step)
                    writer.add_scalar("train/learning_rate", self.cfg.learning_rate, step)
                    
                    writer.add_scalar("time/fps", fps, step)
                    
                    writer.add_scalar("env/collision_rate", collision_rate, step)
                    if ep_speeds:
                        writer.add_scalar("env/speed", np.mean(ep_speeds), step)
                    for key, vals in ep_sub_rewards.items():
                        writer.add_scalar(f"env/reward_{key}", np.mean(vals), step)
                    writer.flush()
                
                # Reset for next episode
                obs, _ = env.reset()
                ep_reward = 0.0
                ep_len = 0
                ep_speeds = []
                ep_sub_rewards = {}
                        
        # End of training
        self.save_checkpoint(tag="final_episodic")
        if writer is not None:
            writer.flush()
            writer.close()

    def save(self, path: str) -> None:
        """Interface BaseAgent -> delegates to save_checkpoint with the given path."""
        torch.save({
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str) -> None:
        """Interface BaseAgent -> delegates to load_checkpoint."""
        self.load_checkpoint(path)

    # Internal methods

    def get_epsilon(self) -> float:
        frac = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, obs) -> int:
        """epsilon-greedy (training)."""
        if random.random() < self.get_epsilon():
            return random.randint(0, self.n_actions - 1)
        return self._greedy(obs)

    def select_actions_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """epsilon-greedy vectorized for N envs (train_vectorized)."""
        n = len(obs_batch)
        actions = np.array([random.randint(0, self.n_actions - 1) for _ in range(n)])
        greedy_mask = np.random.rand(n) >= self.get_epsilon()
        if greedy_mask.any():
            obs_t = torch.tensor(obs_batch[greedy_mask],
                                dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions[greedy_mask] = self.q_net(obs_t).argmax(dim=1).cpu().numpy()
        return actions

    def _greedy(self, obs) -> int:
        """Greedy selection over one observation"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # Ensures that the batch dimension respects obs dimension
        if obs_t.dim() < 2:
            obs_t = obs_t.unsqueeze(0)
        elif obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)   # (H, W) → (1, H, W)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=1).item()

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_checkpoint(self, tag: str = "latest") -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.cfg.checkpoint_dir,
                            f"{timestamp}_dqn_highway_{tag}.pt")
        torch.save({
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        return path

    def load_checkpoint(self, path: str, show:bool=True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        if show:
            print(f"Checkpoint loaded from {path} (step {self.global_step})")
