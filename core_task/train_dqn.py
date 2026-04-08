import random
import time
import numpy as np
import torch
import gymnasium as gym
import highway_env  
from tqdm import tqdm
from typing import Optional

from shared_core_config import SHARED_CORE_CONFIG
from agents.dqn_custom import HighwayDQNConfig, DQNAgent

LOG_FREQ = 5_000


def train_highway_dqn(
    cfg: HighwayDQNConfig,
    resume_from: Optional[str] = None,
):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    obs, _ = env.reset(seed=cfg.seed)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(cfg, obs_shape, n_actions)

    # ─── Reprise depuis checkpoint ────────────────────────────────────────────
    start_step = 0
    episode_rewards: list[float] = []
    ep_lengths: list[int] = []
    losses: list[float] = []

    if resume_from is not None:
        agent.load_checkpoint(resume_from)
        start_step = agent.global_step

        # Recharge les métriques accumulées si elles existent
        for fname, target in [
            ("episode_rewards.npy", episode_rewards),
            ("ep_lengths.npy",      ep_lengths),
            ("losses.npy",          losses),
        ]:
            path = f"{cfg.checkpoint_dir}/{fname}"
            try:
                target.extend(np.load(path).tolist())
                print(f"  Métriques chargées : {path} ({len(target)} entrées)")
            except FileNotFoundError:
                print(f"  Métriques absentes ({fname}), on repart de zéro pour celles-ci")

        print(f"\n  Reprise depuis le step {start_step:,} / {cfg.total_timesteps:,}\n")

    remaining_steps = cfg.total_timesteps - start_step
    if remaining_steps <= 0:
        print("  L'agent a déjà atteint total_timesteps. Rien à entraîner.")
        return agent, episode_rewards, losses

    # ─── Accumulateurs épisodiques ────────────────────────────────────────────
    episode_reward = 0.0
    ep_len = 0

    t_start = time.perf_counter()

    print(f"\n{'='*65}")
    print(f"  DQN Training — {cfg.env_id}  |  {cfg.total_timesteps:,} steps  |  seed {cfg.seed}")
    print(f"  Device : {agent.device}  |  Buffer : {cfg.buffer_capacity:,}  |  lr : {cfg.learning_rate}")
    if resume_from:
        print(f"  Résumé depuis   : {resume_from}  (step {start_step:,})")
    print(f"{'='*65}\n")

    pbar = tqdm(
        total=cfg.total_timesteps,
        initial=start_step,
        unit="step",
        dynamic_ncols=True,
        colour="green",
    )

    for step in range(start_step, cfg.total_timesteps):
        agent.global_step = step

        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        episode_reward += reward
        ep_len += 1

        if done:
            episode_rewards.append(episode_reward)
            ep_lengths.append(ep_len)
            episode_reward = 0.0
            ep_len = 0
            obs, _ = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()

        if step % cfg.checkpoint_frequency == 0 and step > start_step:
            ckpt_path = agent.save_checkpoint(tag=f"step{step}")
            tqdm.write(f"  ✔ Checkpoint saved → {ckpt_path}")

        if step % LOG_FREQ == 0 and step > start_step and episode_rewards:
            elapsed = time.perf_counter() - t_start
            sps = (step - start_step + 1) / elapsed

            mean_r    = np.mean(episode_rewards[-20:])
            best_r    = np.max(episode_rewards)
            mean_len  = np.mean(ep_lengths[-20:])
            mean_loss = np.mean(losses[-100:]) if losses else float("nan")

            tqdm.write(
                f"  step {step:>7,} / {cfg.total_timesteps:,} "
                f"| ε {agent.get_epsilon():.3f} "
                f"| R̄₂₀ {mean_r:+.3f}  best {best_r:+.3f} "
                f"| len̄₂₀ {mean_len:5.1f} "
                f"| loss̄ {mean_loss:.4f} "
                f"| buf {len(agent.buffer):>6,} "
                f"| ep #{len(episode_rewards):>5,} "
                f"| {sps:,.0f} sps"
            )

        pbar.update(1)

    pbar.close()
    env.close()

    agent.save_checkpoint(tag="final")

    np.save(f"{cfg.checkpoint_dir}/episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{cfg.checkpoint_dir}/ep_lengths.npy",      np.array(ep_lengths))
    np.save(f"{cfg.checkpoint_dir}/losses.npy",          np.array(losses))

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*65}")
    print(f"  Training complete in {elapsed_total/60:.1f} min")
    print(f"  Episodes         : {len(episode_rewards)}")
    print(f"  Mean reward      : {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Best episode     : {np.max(episode_rewards):.3f}")
    print(f"  Mean ep. length  : {np.mean(ep_lengths):.1f} steps")
    print(f"  Metrics saved to : {cfg.checkpoint_dir}/")
    print(f"{'='*65}\n")

    return agent, episode_rewards, losses


if __name__ == "__main__":
    cfg = HighwayDQNConfig()

    # Entraînement from scratch
    #agent, rewards, losses = train_highway_dqn(cfg)

    # Reprise depuis un checkpoint
    agent, rewards, losses = train_highway_dqn(cfg, resume_from="checkpoints/dqn_highway_step30000.pt")