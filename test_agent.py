import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import argparse
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
    """
    Instancie un DQNAgent depuis un checkpoint .pt.

    DQNAgent attend (cfg, obs_shape, n_actions) — on ne peut pas passer
    directement les espaces gymnasium. On reconstruit un HighwayDQNConfig
    minimal ; les poids sont chargés via load_checkpoint().
    """
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    cfg = HighwayDQNConfig()
    agent = DQNAgent(cfg, obs_shape, n_actions)
    agent.load_checkpoint(model_path)
    return agent


def run_episode(
    agent_type: str = "random",
    render: bool = True,
    model_path: str = None,
) -> tuple[float, int]:
    """
    Exécute un épisode complet et retourne (total_reward, nb_steps).

    Parameters
    ----------
    agent_type : "random" | "dqn_custom" | "sb3"
    render     : active le rendu visuel si True
    model_path : chemin vers le checkpoint (.pt pour dqn_custom, .zip pour sb3)
    """
    render_mode = "human" if render else None
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)

    obs, _ = env.reset()

    if agent_type == "random":
        agent = RandomAgent(action_space=env.action_space,observation_space= env.observation_space,epsilon=None)

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

    while not (done or truncated):
        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        if render:
            env.render()

    env.close()
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
            "Défaut : checkpoints/models/test.pt  (dqn_custom) "
            "ou      checkpoints/models/test.zip  (sb3)."
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
        model_path = None   # random : aucun fichier requis

   
    if args.agent != "random":
        if model_path is None or not os.path.exists(model_path):
            print(f"Modèle non trouvé : {model_path}")
            print("Fournissez --model-path ou lancez d'abord l'entraînement.")
            exit(1)

    print(f"Agent      : {args.agent}")
    print(f"Épisodes   : {args.episodes}")
    print(f"Rendu      : {'oui' if render else 'non'}")
    if model_path:
        print(f"Checkpoint : {model_path}")

    rewards = []
    steps = []

    pbar = tqdm(range(args.episodes), desc="Évaluation", unit="épisode")
    for _ in pbar:
        reward, step = run_episode(
            agent_type=args.agent,
            render=render,
            model_path=model_path,
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