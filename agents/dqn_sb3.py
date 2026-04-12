from agents.base_agent import BaseAgent
from stable_baselines3 import DQN
from callbacks import HighwayMetricsCallback
from stable_baselines3.common.callbacks import CheckpointCallback

class SB3DQNAgent(BaseAgent):
    def __init__(self, cfg=None, env=None, model_path=None, tensorboard_log="results/logs/dqn_sb3/", **kwargs):
        """
        Initialise l'agent SB3 en utilisant la même config que les agents custom.
        """
        if model_path is not None:
            self.model = DQN.load(model_path, env=env)
        elif cfg is not None and env is not None:
            # Traduction de ta dataclass vers le format SB3
            sb3_params = {
                "learning_rate": cfg.learning_rate,
                "buffer_size":   cfg.buffer_capacity,
                "learning_starts": cfg.learning_starts,
                "batch_size":    cfg.batch_size,
                "tau":           1.0,  # SB3 utilise tau pour les updates soft, 1.0 = hard update
                "gamma":         cfg.gamma,
                "train_freq":    cfg.train_frequency,
                "gradient_steps": 1,
                "target_update_interval": cfg.target_update_frequency,
                "exploration_fraction": cfg.epsilon_decay_steps / cfg.total_timesteps,
                "exploration_initial_eps": cfg.epsilon_start,
                "exploration_final_eps": cfg.epsilon_end,
                "policy_kwargs": dict(net_arch=cfg.hidden_dims),
                "tensorboard_log": tensorboard_log,
                "verbose": 1,
            }
            self.model = DQN("MlpPolicy", env, **sb3_params)
        else:
            raise ValueError(
                "Il faut fournir soit 'model_path', soit 'cfg' ET 'env'.")
            
    def act(self, obs, epsilon=None):
        """
        Sélectionne une action.
        SB3 gère l'exploration en interne pendant l'entraînement.
        En phase d'utilisation (inférence), on utilise deterministic=True.
        """
        # La méthode predict de SB3 renvoie un tuple (action, state)
        action, _states = self.model.predict(obs, deterministic=True)
        return int(action)

    def update(self, obs, action, reward, terminated, next_obs):
        """
        No-op pour SB3. 
        SB3 gère le Replay Buffer et la descente de gradient en interne 
        pendant l'appel à model.learn().
        """
        pass

    def train(self, env, num_episodes=500, seed=None, log_dir=None, run_name=None):
        """
        Lance l'entraînement.
        Attention: SB3 compte en 'timesteps' et non en épisodes.
        """
        # 30 steps est durée max d'un episode dans la config de test
        total_timesteps = num_episodes * 30
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,           # Sauvegarde tous les 10k steps
            save_path="./logs/checkpoints/",
            name_prefix="sb3_model_opti"
        )
        self.model.learn(total_timesteps=total_timesteps, 
                        tb_log_name=run_name or "SB3_Run",
                        callback=[HighwayMetricsCallback(),
                                checkpoint_callback],
                        reset_num_timesteps=False)

    def save(self, path):
        """Sauvegarde l'archive .zip du modèle."""
        self.model.save(path)

    def load(self, path):
        """Charge l'archive .zip du modèle."""
        current_env = getattr(self.model, "env", None) if hasattr(self, "model") else None
        self.model = DQN.load(path, env=current_env)
