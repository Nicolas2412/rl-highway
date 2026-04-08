# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import gymnasium as gym
from agents.base_agent import BaseAgent
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from shared_core_config import DQN_CUSTOM_PARAMS

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        action_space,
        observation_space,
        params=None,
    ):
        p = params if params is not None else DQN_CUSTOM_PARAMS
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = p.get("gamma", 0.99)
        self.batch_size = p.get("batch_size", 16)
        self.buffer_capacity = p.get("buffer_capacity", 15_000)
        self.update_target_every = p.get("update_target_every", 100)
        self.epsilon_start = p.get("epsilon_start", 1)
        self.decrease_epsilon_factor = p.get("decrease_epsilon_factor", 100)
        self.epsilon_min = p.get("epsilon_min", 0.05)
        self.learning_rate = p.get("learning_rate", 5e-4)
        self.hidden_size = p.get("hidden_size", 128)
        
        self.reset()
        
    @property
    def needs_training(self):
        return True

    def update(self, state, action, reward, terminated, next_state):

        # add data to replay buffer
        self.buffer.push(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward], dtype=torch.float32),        # ← add dtype
            torch.tensor([terminated], dtype=torch.float32),    # ← change to float32 too
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        transitions = self.buffer.sample(self.batch_size)

        (
            state_batch,
            action_batch,
            reward_batch,
            terminated_batch,
            next_state_batch,
        ) = tuple([torch.cat(data) for data in zip(*transitions)])

        values = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch

        loss = self.loss_function(values, targets.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.detach().numpy()

    def act(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.get_q(state))
        
    def get_q(self, state):
      with torch.no_grad():
          return self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )

    def reset(self):
        hidden_size = self.hidden_size

        obs_size = np.prod(self.observation_space.shape)
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_size, hidden_size, n_actions)
        self.target_net = Net(obs_size, hidden_size, n_actions)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
        
    def train(self, env, num_episodes=500, seed=None, log_dir="results/logs/dqn_custom", run_name="DQN_Custom"):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

        episode_count = 0
        collision_count = 0
        
        start_time = time.time()
        total_steps = 0
    
        obs, _ = env.reset(seed=seed)
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done, truncated = False, False
            total_reward, steps, crashed = 0, 0, False
            step_start = time.time()
            while not (done or truncated):
                action = self.act(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                
                #Log Tensorboard par step
                if "speed" in info:
                    writer.add_scalar("env/speed", info["speed"], total_steps)
                if "rewards" in info:
                    for key, val in info["rewards"].items():
                        writer.add_scalar(f"env/reward_{key}", val, total_steps)
                
                loss = self.update(obs, action, reward, done or truncated, next_obs)
                if loss is not None and loss != np.inf:
                    writer.add_scalar("train/loss", loss, total_steps)
                    
                if total_steps % 10 == 0:
                    writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], total_steps)
                    
                    elapsed_time = time.time() - start_time
                    fps = total_steps / elapsed_time if elapsed_time > 0 else 0
                    writer.add_scalar("time/fps", fps, total_steps)
                    
                    writer.add_scalar("rollout/exploration_rate", self.epsilon, total_steps)
                    
                obs = next_obs
                total_reward += reward
                steps += 1
                total_steps += 1
                if info.get("crashed", False):
                    crashed = True
            
            #Log Tensoboard par episode
            episode_count += 1
            if crashed:
                collision_count += 1
                
            writer.add_scalar("rollout/ep_rew_mean", total_reward, total_steps)
            writer.add_scalar("rollout/ep_len_mean", steps, total_steps)
            writer.add_scalar("env/collision_rate", collision_count / episode_count, total_steps)
            writer.add_scalar("env/epsilon", self.epsilon, total_steps)
        
        writer.close()

    def _plot_training(self, rewards, lengths, collisions, path):
        window = 50

        def smooth(data):
            return [np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))]

        metrics = [
            (rewards,    "Episode Reward",    "Total Reward", "steelblue"),
            (lengths,    "Episode Length",    "Steps",        "darkorange"),
            (collisions, "Collision Rate",    "Collision Rate","crimson"),
        ]

        base, ext = os.path.splitext(path)
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        for data, title, ylabel, color in metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(data, alpha=0.2, color=color)
            ax.plot(smooth(data), label=f"Smoothed (w={window})", color=color)
            ax.set_title(f"DQN Custom — {title}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            if ylabel == "Collision Rate":
                ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()

            slug = title.lower().replace(" ", "_")
            save_path = f"{base}_{slug}{ext}"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved: {save_path}")

    def save(self, path):
        import os, torch
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        import torch
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        self.epsilon = self.epsilon_min
