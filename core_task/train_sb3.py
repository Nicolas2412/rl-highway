import argparse
import os
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG


def make_env():
    """Crée une instance de l'environnement avec la config partagée."""
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset()
        return env
    return _init


def train(save_path, steps=5000, num_envs=4):
    print(f"Initialisation de {num_envs} environnements pour {steps} steps...")

    # Utilisation d'environnements vectorisés pour accélérer l'entraînement
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=".results/logs/sb3/"
    )

    print(f"Début de l'entraînement : {save_path.split('/')[-1]}")
    model.learn(total_timesteps=steps)

    model.save(save_path)
    print(f"Modèle sauvegardé sous : {save_path}.zip")

    env.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Entraînement SB3 DQN")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Nombre total de steps")
    parser.add_argument("--envs", type=int, default=4,
                        help="Nombre d'environnements parallèles")
    parser.add_argument("--name", type=str,
                        default="sb3_highway_model", help="Nom du modèle")

    args = parser.parse_args()

    # Calcul automatique du chemin de sauvegarde
    save_dir = "results/checkpoints/sb3"
    save_path = os.path.join(save_dir, args.name)

    # Vérification de l'existence du modèle (SB3 ajoute .zip à la fin)
    if os.path.exists(f"{save_path}.zip"):
        print(
            f"Le modèle '{save_path}.zip' existe déjà. Entraînement annulé.")
        exit(1)

    # Création du dossier de sauvegarde si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    train(save_path, steps=args.steps, num_envs=args.envs)
