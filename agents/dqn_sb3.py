from agents.base_agent import BaseAgent
from stable_baselines3 import DQN
from callbacks import HighwayMetricsCallback

class SB3DQNAgent(BaseAgent):
    def __init__(self, env=None, model_path=None, tensorboard_log="results/logs/dqn_sb3/", **kwargs):
        """
        Initialise l'agent SB3.
        Soit on crée un nouveau modèle depuis l'environnement, 
        soit on charge un modèle pré-entraîné.
        """
        if model_path is not None:
            self.model = DQN.load(model_path, env=env)
        elif env is not None:
            # Transfert des hyperparamètres SB3 via kwargs (learning_rate, buffer_size, etc.)
            self.model = DQN("MlpPolicy", env, tensorboard_log=tensorboard_log, **kwargs)
        else:
            raise ValueError("Il faut fournir soit un 'env' soit un 'model_path'.")

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
        
        self.model.learn(total_timesteps=total_timesteps, 
                        tb_log_name=run_name or "SB3_Run",
                        callback= HighwayMetricsCallback(),
                        reset_num_timesteps=False)

    def save(self, path):
        """Sauvegarde l'archive .zip du modèle."""
        self.model.save(path)

    def load(self, path):
        """Charge l'archive .zip du modèle."""
        self.model = DQN.load(path)