import os
import sys
import time
# Permet d'importer le fichier de config depuis le dossier parent
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG

import argparse
import os
import gymnasium as gym

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor 

class HighwayMetricsCallback(BaseCallback):
    """
    Callback personnalisé pour logger toutes les métriques de highway-env.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.collision_count = 0

    def _on_step(self) -> bool:
        # 1Métriques par step (Vitesse, Récompenses détaillées)
        for info in self.locals["infos"]:
            # Log de la vitesse brute
            if "speed" in info:
                self.logger.record("env/speed", info["speed"])
            
            # Log de la décomposition de la récompense (vitesse, collision, file)
            if "rewards" in info:
                for key, val in info["rewards"].items():
                    # On crée des courbes : env/reward_collision, env/reward_high_speed, etc.
                    self.logger.record(f"env/reward_{key}", val)

        # Métriques par épisode (Collision Rate)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                # On vérifie si l'épisode s'est fini par un crash
                if self.locals["infos"][i].get("crashed", False):
                    self.collision_count += 1
                
                # Enregistre le taux de collision global depuis le début
                collision_rate = self.collision_count / self.episode_count
                self.logger.record("env/collision_rate", collision_rate)
        
        return True

def make_env():
    """Crée une instance de l'environnement avec la config partagée."""
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID)
        
        train_config = SHARED_CORE_CONFIG.copy()
        
        train_config["observation"].update({"vehicles_count": 5})
        
        # On ajuste la config pour un meilleur entraînement
        train_config.update({
            "vehicles_count": 25,      
            "collision_reward": -10.0,
        })
        
        env.unwrapped.configure(train_config)
        env = Monitor(env)
        return env
    return _init


def train(save_path, steps=5000, num_envs=4, save_freq=10000):
    print(f"Initialisation de {num_envs} env pour {steps} steps...")

    # Utilisation d'environnements vectorisés pour accélérer l'entraînement
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # Configuration du Callback de sauvegarde
    # save_freq est par rapport au nombre de steps cumulées sur tous les envs.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // num_envs, 1),
        save_path=f"results/checkpoints/sb3/{save_path.split('/')[-1]}",
        name_prefix="cp",
    )
    
    metrics_callback = HighwayMetricsCallback()
    
    callback = CallbackList([checkpoint_callback, metrics_callback])

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="results/logs/sb3/",
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
    )

    print(f"Début de l'entraînement : {save_path.split('/')[-1]}")
    model.learn(total_timesteps=steps, callback=callback,
        tb_log_name=save_path.split('/')[-1])

    model.save(save_path)
    print(f"Modèle sauvegardé sous : {save_path}.zip")

    env.close()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Entraînement SB3 DQN")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Nombre total de steps")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Fréquence de sauvegarde (en steps)")
    parser.add_argument("--envs", type=int, default=1,
                        help="Nombre d'environnements parallèles")
    parser.add_argument("--name", type=str,
                        default=None, help="Nom du modèle")

    args = parser.parse_args()

    # Calcul automatique du chemin de sauvegarde
    save_dir = f"results/models/sb3/"
    os.makedirs(save_dir, exist_ok=True)
    
    date_str = time.strftime("%d%b_%Hh%M")
    if args.name:
        model_name = f"{args.name}_{date_str}"
    else:
        model_name = f"SB3_steps{args.steps}_{date_str}"
        
    save_path = os.path.join(save_dir, model_name)

    # Vérification de l'existence du modèle (SB3 ajoute .zip à la fin)
    if os.path.exists(f"{save_path}.zip"):
        print(
            f"Le modèle '{save_path}.zip' existe déjà. Entraînement annulé.")
        exit(1)

    # Création du dossier de sauvegarde si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    train(save_path, steps=args.steps, num_envs=args.envs, save_freq=args.save_freq)
