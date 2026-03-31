import gymnasium as gym
import highway_env
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG


def create_core_env(render_mode=None):
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return env


if __name__ == "__main__":
    env = create_core_env(render_mode="rgb_array")
    obs, info = env.reset()

    print("Environnement initialisé avec succès.")
    print("Type d'observation :", SHARED_CORE_CONFIG["observation"]["type"])
    print("Type d'action :", SHARED_CORE_CONFIG["action"]["type"])
    print("Forme de l'observation :", obs.shape)

    env.close()
