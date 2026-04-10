from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Un agent basique qui sélectionne des actions de manière purement aléatoire.
    """
    def __init__(self, action_space, observation_space, **kwargs):
        self.action_space = action_space

    def act(self, obs,epsilon=None)-> int:
        """
        Renvoie une action aléatoire tirée de l'espace d'actions.
        L'observation est ignorée.
        """
        action = self.action_space.sample()
        return action
