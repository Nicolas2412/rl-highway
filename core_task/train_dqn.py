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
"""

import random
import time
import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401
from tqdm import tqdm
from typing import Optional

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.dqn_custom import HighwayDQNConfig, DQNAgent

LOG_FREQ = 5_000


def make_env(seed_offset: int = 0):
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset(seed=seed_offset)
        return env
    return _init


def train_vectorized(
    cfg: HighwayDQNConfig,
    num_envs: int = 2,
    resume_from: Optional[str] = None,
):
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

    start_step = 0
    episode_rewards: list[float] = []
    ep_lengths: list[int] = []
    losses: list[float] = []

    if resume_from is not None:
        agent.load_checkpoint(resume_from)
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

    # Accumulateurs par env
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)

    t_start = time.perf_counter()

    pbar = tqdm(total=cfg.total_timesteps, initial=start_step,
                unit="step", dynamic_ncols=True, colour="cyan")

    for step in range(start_step, cfg.total_timesteps):
        agent.global_step = step

        actions = agent.select_actions_batch(obs)
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)

        # Pousser N transitions dans le buffer
        for i in range(num_envs):
            agent.buffer.push(
                obs[i], actions[i], rewards[i], next_obs[i],
                float(terminated[i]),      # terminated uniquement pour le masque
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

        # Mise à jour du réseau
        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()

        if step % cfg.checkpoint_frequency == 0 and step > start_step:
            ckpt = agent.save_checkpoint(tag=f"step{step}")
            tqdm.write(f"  Checkpoint → {ckpt}")

        if step % LOG_FREQ == 0 and step > start_step and episode_rewards:
            elapsed = time.perf_counter() - t_start
            sps = (step - start_step + 1) / elapsed
            mean_r    = np.mean(episode_rewards[-20:])
            mean_loss = np.mean(losses[-100:]) if losses else float("nan")
            tqdm.write(
                f"  step {step:>7,} | ε {agent.get_epsilon():.3f} "
                f"| R̄₂₀ {mean_r:+.3f} | loss̄ {mean_loss:.4f} "
                f"| buf {len(agent.buffer):>6,} | ep #{len(episode_rewards):>5,} "
                f"| {sps:,.0f} sps"
            )

        pbar.update(1)

    pbar.close()
    envs.close()
    agent.save_checkpoint(tag="final")

    np.save(f"{cfg.checkpoint_dir}/episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{cfg.checkpoint_dir}/ep_lengths.npy",      np.array(ep_lengths))
    np.save(f"{cfg.checkpoint_dir}/losses.npy",          np.array(losses))

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTraining terminé en {elapsed_total/60:.1f} min")
    print(f"Épisodes : {len(episode_rewards)} | "
          f"R̄ : {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f} | "
          f"Best : {np.max(episode_rewards):.3f}")

    return agent, episode_rewards, losses


if __name__ == "__main__":

    cfg = HighwayDQNConfig(total_timesteps=200_000)    
    train_vectorized(cfg, num_envs=2, resume_from=None)
