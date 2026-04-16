"""
Usage :

python -m test_agent --agent dqn_custom  --model-path "path_to_model" --episodes 1
python -m test_agent --agent dqn_custom  --model-path "path_to_model" --episodes 1 --save-gif

python -m test_agent --agent dqn_custom  --model-path "checkpoints/old-runs/20260410-112718_dqn_highway_final_episodic.pt" --episodes 1 --save-gif



"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import argparse
import torch
import os
import numpy as np
import gymnasium as gym
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from agents.random_agent import RandomAgent
from agents.dqn_sb3 import SB3DQNAgent
from agents.dqn_custom import DQNAgent, HighwayDQNConfig
import highway_env  # noqa: F401
from tqdm import tqdm


def make_env():
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset()
        return env
    return _init


def _build_dqn_agent(env: gym.Env, model_path: str) -> DQNAgent:
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Inférer hidden_dims depuis le checkpoint pour la rétrocompatibilité
    ckpt = torch.load(model_path, map_location="cpu")
    keys = list(ckpt["q_net"].keys())
    # Chaque couche Linear contribue weight + bias → poids pairs = couches
    # Les couches cachées sont toutes sauf la dernière
    weight_keys = [k for k in keys if k.endswith(".weight")]
    hidden_dims = []
    for wk in weight_keys[:-1]:           # exclut la couche de sortie
        hidden_dims.append(ckpt["q_net"][wk].shape[0])

    cfg = HighwayDQNConfig(hidden_dims=hidden_dims)
    agent = DQNAgent(cfg, obs_shape, n_actions)
    agent.load_checkpoint(model_path)
    return agent

def _gif_path_for_model(model_path: str, agent_type: str, global_step: int | None) -> str:
    """
    Construit le chemin de sortie du GIF :
      <dossier_du_modèle>/<nom_du_dossier>_step<global_step>.gif

    Si global_step est None (agent SB3 ou random), on n'ajoute pas le suffixe step.
    """
    model_dir  = os.path.dirname(os.path.abspath(model_path))
    folder_name = os.path.basename(model_dir)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    base = f"{folder_name}_{stem}"
    if global_step is not None:
        base = f"{base}_step{global_step}"
    return os.path.join(model_dir, f"{base}.gif")


def run_episode(
    agent_type: str = "random",
    render: bool = True,
    model_path: str = None,
    save_gif: bool = False,
) -> tuple[float, int]:
    """
    Exécute un épisode complet et retourne (total_reward, nb_steps).

    Parameters
    ----------
    agent_type : "random" | "dqn_custom" | "sb3"
    render     : active le rendu visuel si True
    model_path : chemin vers le checkpoint (.pt pour dqn_custom, .zip pour sb3)
    save_gif   : enregistre un GIF de l'épisode dans le dossier du modèle
    """
    # Quand on enregistre un GIF on a besoin des frames en tableau numpy,
    # ce qui requiert render_mode="rgb_array". Ces deux modes sont exclusifs :
    # on ne peut pas avoir "human" et "rgb_array" simultanément.
    if save_gif:
        render_mode = "rgb_array"
        render = False  # le rendu "human" est désactivé dans ce cas
    else:
        render_mode = "human" if render else None

    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)

    obs, _ = env.reset()

    if agent_type == "random":
        agent = RandomAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            epsilon=None,
        )

    elif agent_type == "dqn_custom":
        if model_path is None:
            raise ValueError(
                "dqn_custom requiert --model-path pointant vers un fichier .pt"
            )
        agent = _build_dqn_agent(env, model_path)

    elif agent_type == "sb3":
        if model_path is None:
            raise ValueError(
                "sb3 requiert --model-path pointant vers un fichier .zip"
            )
        agent = SB3DQNAgent(
            model_path=model_path,
            action_space=env.action_space,
            determistic=True,
        )

    else:
        raise ValueError(f"Agent inconnu : {agent_type}")

    done = False
    truncated = False
    total_reward = 0.0
    step = 0
    frames: list[np.ndarray] = []

    while not (done or truncated):
        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        if save_gif:
            frame = env.render()  # renvoie un np.ndarray (H, W, 3) en mode rgb_array
            frames.append(frame)
        elif render:
            env.render()

    env.close()

    if save_gif and frames and model_path is not None:
        import imageio

        global_step = getattr(agent, "global_step", None)
        gif_path = _gif_path_for_model(model_path, agent_type, global_step)
        imageio.mimsave(gif_path, frames, fps=4, loop=0)
        print(f"GIF enregistré : {gif_path}")

    return total_reward, step


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluation des agents sur Highway-env"
    )
    parser.add_argument(
        "--agent", type=str, default="random",
        choices=["random", "sb3", "dqn_custom"],
        help="Type d'agent à évaluer.",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Nombre d'épisodes d'évaluation.",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Désactive le rendu visuel.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help=(
            "Chemin vers le checkpoint. "
            "Défaut : results/models/test.pt  (dqn_custom) "
            "ou      results/models/test.zip  (sb3)."
        ),
    )
    parser.add_argument(
        "--save-gif", action="store_true",
        help=(
            "Enregistre un GIF du premier épisode dans le dossier du modèle. "
            "Désactive automatiquement le rendu visuel (modes exclusifs)."
        ),
    )

    args = parser.parse_args()
    render = not args.no_render

    if args.model_path is not None:
        model_path = args.model_path
    elif args.agent == "dqn_custom":
        model_path = "results/models/test.pt"
    elif args.agent == "sb3":
        model_path = "results/models/test.zip"
    else:
        model_path = None

    if args.agent != "random":
        if model_path is None or not os.path.exists(model_path):
            print(f"Modèle non trouvé : {model_path}")
            print("Fournissez --model-path ou lancez d'abord l'entraînement.")
            exit(1)

    print(f"Agent      : {args.agent}")
    print(f"Épisodes   : {args.episodes}")
    print(f"Rendu      : {'oui' if render and not args.save_gif else 'non'}")
    print(f"Save GIF   : {'oui' if args.save_gif else 'non'}")
    if model_path:
        print(f"Checkpoint : {model_path}")

    rewards = []
    steps = []

    pbar = tqdm(range(args.episodes), desc="Évaluation", unit="épisode")
    for i in pbar:
        # On n'enregistre le GIF que sur le premier épisode pour ne pas
        # écraser le fichier à chaque itération. Adapter si besoin.
        episode_save_gif = args.save_gif and (i == 0)
        reward, step = run_episode(
            agent_type=args.agent,
            render=render,
            model_path=model_path,
            save_gif=episode_save_gif,
        )
        rewards.append(reward)
        steps.append(step)
        pbar.set_postfix({
            "Last_Rew": f"{reward:.2f}",
            "Steps":    step,
            "Avg_Rew":  f"{np.mean(rewards):.2f}",
        })

    if args.episodes > 0:
        print("\n--- Statistiques Globales ---")
        print(f"Récompense moyenne : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Durée moyenne      : {np.mean(steps):.1f} étapes")