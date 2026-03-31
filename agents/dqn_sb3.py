from stable_baselines3 import DQN

class SB3Agent:
    def __init__(self, model_path=None, action_space=None):
        """
        Initialise l'agent SB3. 
        Si model_path est fourni, le modèle est chargé depuis le disque.
        """
        self.action_space = action_space
        self.model = None

        if model_path:
            self.load(model_path)

    def load(self, model_path):
        """Charge un modèle DQN entraîné avec Stable-Baselines3."""
        self.model = DQN.load(model_path)

    def act(self, observation):
        """
        Prédit l'action à partir de l'observation.
        SB3 renvoie (action, _states), on ne garde que l'action.
        """
        if self.model is not None:
            # deterministic=True est crucial pour l'évaluation [cite: 42]
            action, _states = self.model.predict(
                observation, deterministic=True)
            return action
        else:
            # Fallback si aucun modèle n'est chargé
            return self.action_space.sample()
