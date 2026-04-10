"""
Vectorized training entrypoint for simple DQN + Prioritized Experience Replay.

This intentionally mirrors core_task/train_dqn.py so the PER variant remains
isolated from the vanilla DQN pipeline.
"""

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from typing import Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agents.dqn_per import HighwayPERConfig, PERDQNAgent
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID

LOG_FREQ = 5_000
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(WORKING_DIR, f"checkpoints/per_dqn_{TIMESTAMP}")
REGISTRY_PATH = os.path.join(WORKING_DIR, "checkpoints/per_runs_registry.jsonl")


def cfg_to_dict(cfg: HighwayPERConfig) -> dict:
    return dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)


def register_run_start(cfg: HighwayPERConfig, num_envs: int, run_id: str) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    entry = {
        "run_id": run_id,
        "algorithm": "per_dqn",
        "status": "running",
        "started_at": TIMESTAMP,
        "ended_at": None,
        "num_envs": num_envs,
        "checkpoint_dir": CHECKPOINT_DIR,
        "hyperparameters": cfg_to_dict(cfg),
        "results": None,
    }
    with open(REGISTRY_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def register_run_end(run_id: str, episode_rewards: list[float], final_checkpoint: str) -> None:
    if not os.path.exists(REGISTRY_PATH):
        return

    with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    updated_lines = []
    for line in lines:
        entry = json.loads(line)
        if entry["run_id"] == run_id:
            entry["status"] = "done"
            entry["ended_at"] = time.strftime("%Y%m%d-%H%M%S")
            entry["final_checkpoint"] = final_checkpoint
            entry["results"] = {
                "n_episodes": len(episode_rewards),
                "mean_reward": round(float(np.mean(episode_rewards)), 4) if episode_rewards else None,
                "std_reward": round(float(np.std(episode_rewards)), 4) if episode_rewards else None,
                "best_reward": round(float(np.max(episode_rewards)), 4) if episode_rewards else None,
                "worst_reward": round(float(np.min(episode_rewards)), 4) if episode_rewards else None,
            }
        updated_lines.append(json.dumps(entry) + "\n")

    with open(REGISTRY_PATH, "w", encoding="utf-8") as handle:
        handle.writelines(updated_lines)


def make_env(seed_offset: int = 0):
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset(seed=seed_offset)
        return env

    return _init


def train_vectorized(
    cfg: HighwayPERConfig,
    num_envs: int = 2,
    resume_from: Optional[str] = None,
):
    run_id = f"per_dqn_{TIMESTAMP}"
    cfg.checkpoint_dir = CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = gym.vector.AsyncVectorEnv([make_env(cfg.seed + idx) for idx in range(num_envs)])
    observations, _ = envs.reset(seed=cfg.seed)

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    agent = PERDQNAgent(cfg, obs_shape, n_actions)
    print(f"Device : {agent.device} | Envs paralleles : {num_envs}")
    print(f"Run ID : {run_id}")
    print(f"Checkpoints -> {CHECKPOINT_DIR}")
    print(f"Registry    -> {REGISTRY_PATH}")

    register_run_start(cfg, num_envs, run_id)

    start_step = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []

    if resume_from is not None:
        resume_path = resume_from if os.path.isabs(resume_from) else os.path.join(CHECKPOINT_DIR, resume_from)
        agent.load_checkpoint(resume_path)
        start_step = agent.global_step
        for file_name, target in [
            ("episode_rewards.npy", episode_rewards),
            ("ep_lengths.npy", episode_lengths),
            ("losses.npy", losses),
        ]:
            path = os.path.join(CHECKPOINT_DIR, file_name)
            try:
                target.extend(np.load(path).tolist())
            except FileNotFoundError:
                pass

    remaining = cfg.total_timesteps - start_step
    if remaining <= 0:
        print("total_timesteps deja atteint.")
        envs.close()
        return agent, episode_rewards, losses

    current_rewards = np.zeros(num_envs, dtype=np.float32)
    current_lengths = np.zeros(num_envs, dtype=np.int32)
    start_time = time.perf_counter()

    pbar = tqdm(total=cfg.total_timesteps, initial=start_step, unit="step", dynamic_ncols=True, colour="cyan")

    for step in range(start_step, cfg.total_timesteps):
        agent.global_step = step

        actions = agent.select_actions_batch(observations)
        next_observations, rewards, terminated, truncated, _ = envs.step(actions)

        for idx in range(num_envs):
            agent.buffer.push(
                observations[idx],
                actions[idx],
                rewards[idx],
                next_observations[idx],
                float(terminated[idx]),
            )

        observations = next_observations
        current_rewards += rewards
        current_lengths += 1

        done_mask = terminated | truncated
        for idx in range(num_envs):
            if done_mask[idx]:
                episode_rewards.append(float(current_rewards[idx]))
                episode_lengths.append(int(current_lengths[idx]))
                current_rewards[idx] = 0.0
                current_lengths[idx] = 0

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()

        if cfg.checkpoint_frequency > 0 and step % cfg.checkpoint_frequency == 0:
            agent.save_checkpoint(tag=f"step{step}")
            np.save(os.path.join(CHECKPOINT_DIR, "episode_rewards.npy"), np.array(episode_rewards))
            np.save(os.path.join(CHECKPOINT_DIR, "ep_lengths.npy"), np.array(episode_lengths))
            np.save(os.path.join(CHECKPOINT_DIR, "losses.npy"), np.array(losses))

        if step % LOG_FREQ == 0 and episode_rewards:
            recent_rewards = episode_rewards[-50:]
            elapsed = time.perf_counter() - start_time
            sps = (step - start_step) / max(elapsed, 1e-6)
            pbar.set_postfix(
                {
                    "ep": len(episode_rewards),
                    "r_mean": f"{np.mean(recent_rewards):.2f}",
                    "r_std": f"{np.std(recent_rewards):.2f}",
                    "eps": f"{agent.get_epsilon():.3f}",
                    "beta": f"{agent.get_beta():.3f}",
                    "sps": f"{sps:.0f}",
                }
            )

        pbar.update(1)

    pbar.close()
    envs.close()

    final_checkpoint = agent.save_checkpoint(tag="final")
    np.save(os.path.join(CHECKPOINT_DIR, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(CHECKPOINT_DIR, "ep_lengths.npy"), np.array(episode_lengths))
    np.save(os.path.join(CHECKPOINT_DIR, "losses.npy"), np.array(losses))

    register_run_end(run_id, episode_rewards, final_checkpoint)

    total_time = time.perf_counter() - start_time
    print(f"\nEntrainement termine en {total_time:.1f}s")
    print(f"  Episodes   : {len(episode_rewards)}")
    if episode_rewards:
        print(f"  Reward mean: {np.mean(episode_rewards[-50:]):.3f}")
        print(f"  Reward std : {np.std(episode_rewards[-50:]):.3f}")
    print(f"  Checkpoint : {final_checkpoint}")

    return agent, episode_rewards, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PER DQN agent on highway-v0")
    parser.add_argument("--steps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--envs", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path or name to resume from")
    parser.add_argument("--double-dqn", action="store_true", help="Enable Double DQN target selection")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER alpha exponent")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="Initial PER beta")
    parser.add_argument("--per-beta-end", type=float, default=1.0, help="Final PER beta")
    parser.add_argument("--per-epsilon", type=float, default=1e-5, help="Priority epsilon")
    args = parser.parse_args()

    config = HighwayPERConfig(
        seed=args.seed,
        total_timesteps=args.steps,
        double_dqn=args.double_dqn,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_epsilon=args.per_epsilon,
    )
    train_vectorized(config, num_envs=args.envs, resume_from=args.resume_from)
