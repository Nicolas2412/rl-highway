"""
Classe pour le DQN
espsilon-decay linéaire
Inclus une option pour un double DQN

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


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class HighwayDQNConfig:
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
    checkpoint_dir: str = "../rl-highway/checkpoints"
    checkpoint_frequency: int = 10_000
    
    def __init__(self,env_id = "highway-v0",
                    seed = 42,
                    hidden_dims = [256, 256],
                    total_timesteps = 200_000,
                    learning_rate = 5e-4,
                    gamma = 0.9,
                    batch_size = 32,
                    buffer_capacity = 15_000,
                    learning_starts = 200,
                    train_frequency = 1,
                    target_update_frequency = 50,
                    epsilon_start = 1.0,
                    epsilon_end = 0.01,
                    epsilon_decay_steps = 100_000,
                    double_dqn = False,
                    checkpoint_dir = "../rl-highway/checkpoints",
                    checkpoint_frequency = 10_000):
        self.env_id = env_id
        self.seed = seed
        self.hidden_dims = hidden_dims
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.double_dqn = double_dqn


# ─── Q-Network ────────────────────────────────────────────────────────────────

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
        # Aplatit toutes les dimensions sauf le batch
        return self.net(x.flatten(start_dim=1) if x.dim() > 1 else x.flatten())


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, terminated):
        self.buffer.append((
            np.array(s, dtype=np.float32),
            int(a),
            float(r),
            np.array(s_, dtype=np.float32),
            float(terminated),   # terminated uniquement, pas done (pour le bootstrap)
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


# ─── Agent ────────────────────────────────────────────────────────────────────

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

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def act(self, obs, epsilon=None) -> int:
        """Sélection gloutonne (évaluation). epsilon est ignoré."""
        return self._greedy(obs)

    def update(self, obs=None, action=None, reward=None,
               terminated=None, next_obs=None) -> Optional[float]:
        """
        Tire un mini-batch du buffer et effectue une mise à jour du réseau.
        Les arguments positionnels sont conservés pour respecter la signature
        de BaseAgent ; le push dans le buffer est fait en amont par la boucle
        d'entraînement (train_highway_dqn / train_vectorized / train).
        Retourne la loss ou None si le buffer est trop petit.
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

    def train(self, env, num_episodes: int = 500,
              seed: Optional[int] = None,
              log_dir: Optional[str] = None,
              run_name: Optional[str] = None) -> None:
        """
        Boucle d'entraînement épisodique (interface BaseAgent).
        Utilisée par evaluate_over_seeds ; fonctionne sur un env simple
        (non-vectorisé), contrairement à train_vectorized.

        Les hyperparamètres (lr, gamma, epsilon…) sont lus depuis self.cfg.
        TensorBoard : écrit dans log_dir/run_name si fournis.
        """


        # ── Reproductibilité ─────────────────────────────────────────────────
        _seed = seed if seed is not None else self.cfg.seed
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)

        # ── TensorBoard ──────────────────────────────────────────────────────
        writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            _run_name = run_name or f"dqn_ep_{time.strftime('%Y%m%d-%H%M%S')}"
            writer = SummaryWriter(log_dir=os.path.join(log_dir, _run_name))

        # ── Configuration de l'environnement ─────────────────────────────────
        # L'env est passé par evaluate_over_seeds ; on le configure ici si
        # l'unwrapped expose configure() (highway-env).
        if hasattr(env.unwrapped, "configure"):
            env.unwrapped.configure(SHARED_CORE_CONFIG)

        obs, _ = env.reset(seed=_seed)

        # Compteurs globaux
        step = self.global_step
        episode_rewards: list[float] = []
        losses: list[float] = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=None if ep > 0 else _seed)
            ep_reward = 0.0
            ep_len    = 0
            done = truncated = False

            while not (done or truncated):
                # epsilon-greedy
                action = self.select_action(obs)
                next_obs, reward, done, truncated, _ = env.step(action)

                # Push dans le buffer (terminated uniquement, pas done)
                self.buffer.push(obs, action, reward, next_obs, float(done))
                obs = next_obs
                ep_reward += reward
                ep_len    += 1
                step      += 1
                self.global_step = step

                # Mise à jour réseau
                if step >= self.cfg.learning_starts and step % self.cfg.train_frequency == 0:
                    loss = self.update()
                    if loss is not None:
                        losses.append(loss)
                        if writer is not None:
                            writer.add_scalar("train/loss", loss, step)

                # Sync réseau cible
                if step % self.cfg.target_update_frequency == 0:
                    self.sync_target_network()

                # Checkpoint périodique
                if self.cfg.checkpoint_frequency > 0 and step % self.cfg.checkpoint_frequency == 0:
                    self.save_checkpoint(tag=f"step{step}")

            episode_rewards.append(ep_reward)

            # ── Logging TensorBoard par épisode ──────────────────────────────
            if writer is not None:
                writer.add_scalar("train/episode_reward", ep_reward,  ep)
                writer.add_scalar("train/episode_length", ep_len,     ep)
                writer.add_scalar("train/epsilon",        self.get_epsilon(), step)
                if losses:
                    writer.add_scalar("train/loss_mean_ep",
                                      float(np.mean(losses[-ep_len:])), ep)

        # ── Fin d'entraînement ───────────────────────────────────────────────
        self.save_checkpoint(tag="final_episodic")

        if writer is not None:
            writer.flush()
            writer.close()

    def save(self, path: str) -> None:
        """Interface BaseAgent → délègue à save_checkpoint avec le path fourni."""
        torch.save({
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str) -> None:
        """Interface BaseAgent → délègue à load_checkpoint."""
        self.load_checkpoint(path)

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def get_epsilon(self) -> float:
        frac = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, obs) -> int:
        """epsilon-greedy (entraînement)."""
        if random.random() < self.get_epsilon():
            return random.randint(0, self.n_actions - 1)
        return self._greedy(obs)

    def select_actions_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """epsilon-greedy vectorisé pour N envs (train_vectorized)."""
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
        """Sélection gloutonne sur une seule observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # Assure la présence de la dimension batch quelle que soit la forme d'obs
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

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        print(f"Checkpoint chargé depuis {path} (step {self.global_step})")