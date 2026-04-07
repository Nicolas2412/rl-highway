# agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):

    @abstractmethod
    def act(self, obs, epsilon=None):
        """Select an action given an observation."""
        ...

    def update(self, obs, action, reward, terminated, next_obs):
        """Update the agent (no-op for non-learning agents)."""
        pass

    def train(self, env, num_episodes=500, seed=None, log_dir:str=None, run_name:str=None):
        """Full training loop. No-op for non-learning agents.
        log_dir: Dossier principal pour TensorBoard
        run_name: Nom  de l'expérience 
        """
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