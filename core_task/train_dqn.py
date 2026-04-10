"""
Entraînement DQN vectorisé sur highway-v0.

Principe : N environnements AsyncVectorEnv tournent en parallèle (sous-processus).
À chaque step global, on collecte N transitions simultanément, ce qui :
  - diversifie le replay buffer N fois plus vite ;
  - réduit la corrélation temporelle entre transitions consécutives ;
  - n'augmente PAS le nombre de mises à jour réseau (on reste à 1 update/step global).

AsyncVectorEnv renvoie (obs, reward, terminated, truncated, info).
On stocke `terminated` dans le buffer (masque de bootstrap), et on reset sur
`terminated | truncated`.

python -m core_task.train_dqn
"""

import dataclasses
import json
import random
import time
import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401
from tqdm import tqdm
from typing import Optional
import os
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.dqn_custom import HighwayDQNConfig, DQNAgent

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

LOG_FREQ = 5_000
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Répertoire propre à ce run : checkpoints/dqn_YYYYMMDD-HHMMSS/
CHECKPOINT_DIR = os.path.join(WORKING_DIR, f"checkpoints/dqn_{TIMESTAMP}")

# Registre centralisé de tous les runs, à la racine de checkpoints/
REGISTRY_PATH = os.path.join(WORKING_DIR, "checkpoints/runs_registry.jsonl")


# ---------------------------------------------------------------------------
# Helpers de logging
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg: HighwayDQNConfig) -> dict:
    """Sérialise les hyperparamètres du config en dict JSON-compatible."""
    return dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)


def register_run_start(cfg: HighwayDQNConfig, num_envs: int, run_id: str) -> None:
    """
    Écrit une entrée dans runs_registry.jsonl au démarrage du run.
    Le champ 'status' vaut 'running' ; il sera mis à jour à la fin.
    """
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    entry = {
        "run_id":     run_id,
        "status":     "running",
        "started_at": TIMESTAMP,
        "ended_at":   None,
        "num_envs":   num_envs,
        "checkpoint_dir": CHECKPOINT_DIR,
        "hyperparameters": _cfg_to_dict(cfg),
        "results": None,
    }
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def register_run_end(
    run_id: str,
    episode_rewards: list,
    final_checkpoint: str,
) -> None:
    """
    Relit le registre, met à jour l'entrée du run courant, et réécrit le fichier.
    On relit entièrement pour préserver les entrées des autres runs.
    """
    if not os.path.exists(REGISTRY_PATH):
        return

    with open(REGISTRY_PATH, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        entry = json.loads(line)
        if entry["run_id"] == run_id:
            entry["status"]           = "done"
            entry["ended_at"]         = time.strftime("%Y%m%d-%H%M%S")
            entry["final_checkpoint"] = final_checkpoint
            entry["results"] = {
                "n_episodes":   len(episode_rewards),
                "mean_reward":  round(float(np.mean(episode_rewards)), 4),
                "std_reward":   round(float(np.std(episode_rewards)),  4),
                "best_reward":  round(float(np.max(episode_rewards)),  4),
                "worst_reward": round(float(np.min(episode_rewards)),  4),
            }
        updated_lines.append(json.dumps(entry) + "\n")

    with open(REGISTRY_PATH, "w") as f:
        f.writelines(updated_lines)


def make_env(seed_offset: int = 0):
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset(seed=seed_offset)
        return env
    return _init


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train_vectorized(
    cfg: HighwayDQNConfig,
    num_envs: int = 2,
    resume_from: Optional[str] = None,
):
    # run_id = identifiant unique et lisible du run
    run_id = f"dqn_{TIMESTAMP}"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg.seed + i) for i in range(num_envs)]
    )
    obs, _ = envs.reset(seed=cfg.seed)

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    agent = DQNAgent(cfg, obs_shape, n_actions)
    print(f"Device : {agent.device} | Envs parallèles : {num_envs}")
    print(f"Run ID : {run_id}")
    print(f"Checkpoints → {CHECKPOINT_DIR}")
    print(f"Registre    → {REGISTRY_PATH}")

    # Enregistrement du run dans le registre dès le démarrage
    register_run_start(cfg, num_envs, run_id)

    start_step = 0
    episode_rewards: list[float] = []
    ep_lengths: list[int] = []
    losses: list[float] = []

    if resume_from is not None:
        resume_path = os.path.join(CHECKPOINT_DIR, resume_from)
        agent.load_checkpoint(resume_path)
        start_step = agent.global_step
        for fname, target in [
            ("episode_rewards.npy", episode_rewards),
            ("ep_lengths.npy",      ep_lengths),
            ("losses.npy",          losses),
        ]:
            path = f"{cfg.checkpoint_dir}/{fname}"
            try:
                target.extend(np.load(path).tolist())
            except FileNotFoundError:
                pass

    remaining = cfg.total_timesteps - start_step
    if remaining <= 0:
        print("total_timesteps déjà atteint.")
        envs.close()
        return agent, episode_rewards, losses

    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)

    t_start = time.perf_counter()

    pbar = tqdm(total=cfg.total_timesteps, initial=start_step,
                unit="step", dynamic_ncols=True, colour="cyan")

    for step in range(start_step, cfg.total_timesteps):
        agent.global_step = step

        actions = agent.select_actions_batch(obs)
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)

        for i in range(num_envs):
            agent.buffer.push(
                obs[i], actions[i], rewards[i], next_obs[i],
                float(terminated[i]),
            )

        obs = next_obs
        current_rewards += rewards
        current_lengths += 1

        done_mask = terminated | truncated
        for i in range(num_envs):
            if done_mask[i]:
                episode_rewards.append(current_rewards[i])
                ep_lengths.append(current_lengths[i])
                current_rewards[i] = 0.0
                current_lengths[i] = 0

        if step >= cfg.learning_starts and step % cfg.train_frequency =