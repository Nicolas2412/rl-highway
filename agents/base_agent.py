from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs, epsilon=None) -> int:
        """Select an action given an observation."""
        pass

    def update(self, obs, action, reward, terminated, next_obs) -> Optional[float]:
        """Update the agent (no-op for non-learning agents)."""
        pass

    def train(
        self,
        env,
        num_episodes=500,
        seed=None,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """Full training loop. No-op for non-learning agents."""
        pass

    def save(self, path):
        """Save weights. No-op for agents without weights."""
        pass

    def load(self, path):
        """Load weights. No-op for agents without weights."""
        pass

    @property
    def needs_training(self):
        """Whether this agent requires a training phase."""
        return False
