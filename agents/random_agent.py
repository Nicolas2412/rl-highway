from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Random agent for baseline
    """
    def __init__(self, action_space, observation_space, **kwargs):
        self.action_space = action_space

    def act(self, obs,epsilon=None)-> int:
        """
        Select a completely random action in the action space
        """
        action = self.action_space.sample()
        return action
