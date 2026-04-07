import random
import time
import numpy as np
import torch
import gymnasium as gym
import highway_env  
from tqdm import tqdm

from ..shared_core_config import SHARED_CORE_CONFIG
from ..agents.dqn_custom import HighwayDQNConfig, HighwayDQNAgent

# Log every N steps — tune down for more granularity, up for quieter output
LOG_FREQ = 5_000


# ─── Training loop ────────────────────────────────────────────────────────────

def train_highway_dqn(cfg: HighwayDQNConfig):
    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Build environment — use unwrapped.configure() as required by highway-env
    env = gym.make(cfg.env_id)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    obs, _ = env.reset(seed=cfg.seed)

    obs_shape = env.observation_space.shape  # (5, 5): 5 vehicles × 5 kinematic features
    n_actions = env.action_space.n           # 5 discrete meta-actions

    agent = HighwayDQNAgent(cfg, obs_shape, n_actions)

    # Per-episode accumulators
    episode_reward = 0.0
    ep_len = 0

    # Metric histories (saved at the end for evaluation and plotting)
    episode_rewards: list[float] = []
    ep_lengths: list[int] = []
    losses: list[float] = []

    # Timing for steps/sec estimate
    t_start = time.perf_counter()
    steps_at_last_log = 0

    print(f"\n{'='*65}")
    print(f"  DQN Training — {cfg.env_id}  |  {cfg.total_timesteps:,} steps  |  seed {cfg.seed}")
    print(f"  Device : {agent.device}  |  Buffer : {cfg.buffer_capacity:,}  |  lr : {cfg.learning_rate}")
    print(f"{'='*65}\n")

    pbar = tqdm(
        total=cfg.total_timesteps,
        unit="step",
        dynamic_ncols=True,
        colour="green",
    )

    for step in range(cfg.total_timesteps):
        agent.global_step = step

        # ε-greedy action selection
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition; use `terminated` (not `done`) so the bootstrap
        # target is zero only on true terminal states, not on time-outs.
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

        # Gradient update (delayed start so the buffer has enough samples)
        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        # Hard target-network sync
        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()

        # Periodic checkpoint
        if step % cfg.checkpoint_frequency == 0 and step > 0:
            ckpt_path = agent.save_checkpoint(tag=f"step{step}")
            tqdm.write(f"  ✔ Checkpoint saved → {ckpt_path}")

        # Detailed console log
        if step % LOG_FREQ == 0 and step > 0 and episode_rewards:
            elapsed = time.perf_counter() - t_start
            sps = (step - steps_at_last_log) / max(elapsed - (steps_at_last_log / max(step, 1) * elapsed), 1e-6)
            sps = (step + 1) / elapsed  # simpler: global average steps/sec

            mean_r   = np.mean(episode_rewards[-20:])
            best_r   = np.max(episode_rewards)
            mean_len = np.mean(ep_lengths[-20:])
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

    # Final checkpoint
    agent.save_checkpoint(tag="final")

    # Save metrics as numpy arrays so evaluation scripts can load them directly
    np.save(f"{cfg.checkpoint_dir}/episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{cfg.checkpoint_dir}/ep_lengths.npy", np.array(ep_lengths))
    np.save(f"{cfg.checkpoint_dir}/losses.npy", np.array(losses))

    # Training summary
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


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = HighwayDQNConfig()
    agent, rewards, losses = train_highway_dqn(cfg)