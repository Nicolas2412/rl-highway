class RandomAgent:
    """
    Un agent basique qui sélectionne des actions de manière purement aléatoire.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        """
        Renvoie une action aléatoire tirée de l'espace d'actions.
        L'observation est ignorée.
        """
        return self.action_space.sample()
