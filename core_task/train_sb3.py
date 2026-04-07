import os
import sys
import time
# Permet d'importer le fichier de config depuis le dossier parent
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
# from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from training_config_sb3 import TRAIN_ENV_ID, TRAIN_CONFIG

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


class DecayLaneChangeWrapper(gym.Wrapper):
    """
    Wrapper qui ajoute un bonus aux changements de file, 
    lequel diminue linéairement au fil des étapes.
    """
    def __init__(self, env, total_decay_steps=200000, initial_bonus=0.5, final_bonus=0.05):
        super().__init__(env)
        self.total_decay_steps = total_decay_steps
        self.initial_bonus = initial_bonus
        self.final_bonus = final_bonus
        self.current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if action in [0, 2]:
            fraction = min(1.0, self.current_step / self.total_decay_steps)
            
            # Interpolation linéaire : on part de initial, on arrive à final
            bonus = self.initial_bonus + (self.final_bonus - self.initial_bonus) * fraction
            reward += bonus
            
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # On ne réinitialise PAS current_step ici si on veut que le bonus 
        # diminue sur toute la durée de l'entraînement, pas par épisode.
        return self.env.reset(**kwargs)
    
    
def make_env(total_decay_steps:int):
    def _init():
        env = gym.make(TRAIN_ENV_ID)
        env.unwrapped.configure(TRAIN_CONFIG)
        env = DecayLaneChangeWrapper(env, total_decay_steps=total_decay_steps, initial_bonus=0.05, final_bonus=0.05)
        env.reset() 
        
        return Monitor(env)
    return _init


def train(save_path, steps=5000, num_envs=4, save_freq=10000, load_path=None):
    print(f"Initialisation de {num_envs} env pour {steps} steps...")

    # Utilisation d'environnements vectorisés pour accélérer l'entraînement
    env = SubprocVecEnv([make_env(total_decay_steps = 250000/num_envs) for _ in range(num_envs)])
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // num_envs, 1),
        save_path=f"results/checkpoints/sb3/{save_path.split('/')[-1]}",
        name_prefix="cp",
    )
    
    metrics_callback = HighwayMetricsCallback()
    
    callback = CallbackList([checkpoint_callback, metrics_callback])

    if load_path:
        print(f"Chargement du modèle : {load_path}")
        model = DQN.load(
            load_path, 
            env=env, 
            device="auto",
            custom_objects={
                "exploration_initial_eps": 0.15,  # On met une valeur de départ énorme (virtuelle)
                "exploration_final_eps": 0.05,
                "exploration_fraction":0.5      # On étale le calcul sur 100% de la durée
            }
        )
    else:
        print("Création d'un nouveau modèle...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="results/logs/sb3/",
            exploration_fraction=0.5,
            exploration_final_eps=0.05,
        )

    print(f"Début de l'entraînement : {save_path.split('/')[-1]}")
    model.learn(total_timesteps=steps, callback=callback,
        tb_log_name=save_path.split('/')[-1],
        reset_num_timesteps=(load_path is None))

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
    parser.add_argument("--load", type=str,
                        default=None, help="Chemin du modèle à charger")

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
    
    train(save_path, 
        steps=args.steps, 
        num_envs=args.envs, 
        save_freq=args.save_freq,
        load_path=args.load)
