# utils/callbacks.py
from stable_baselines3.common.callbacks import BaseCallback

class HighwayMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.collision_count = 0

    def _on_step(self) -> bool:
        # On récupère les infos de tous les environnements (SubprocVecEnv)
        for info in self.locals["infos"]:
            if "speed" in info:
                self.logger.record("env/speed", info["speed"])
            
            if "rewards" in info:
                for key, val in info["rewards"].items():
                    self.logger.record(f"env/reward_{key}", val)

        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                if self.locals["infos"][i].get("crashed", False):
                    self.collision_count += 1
                
                # Log du taux de collision global
                self.logger.record("env/collision_rate", self.collision_count / self.episode_count)
        return True