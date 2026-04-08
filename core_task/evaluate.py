import numpy as np
import torch
import gymnasium as gym
import highway_env
from typing import Optional
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.dqn_custom import DQNAgent, HighwayDQNConfig
from tqdm import tqdm

def evaluate_agent(
    agent,
    env_config: dict,
    n_episodes: int = 50,
    seeds: Optional[list] = None,
    is_sb3: bool = False,
) -> dict:
    if seeds is None:
        seeds = list(range(n_episodes))

    assert len(seeds) == n_episodes, "Un seed par épisode"

    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(env_config)   # configure avant reset
    env.reset()                            # applique la config

    episode_rewards = []
    episode_lengths = []
    crash_count = 0

    for ep_idx in tqdm(range(n_episodes), desc="Évaluation", unit="ep"):
        obs, info = env.reset(seed=seeds[ep_idx])
        total_reward = 0.0
        length = 0
        done = False

        while not done:
            if is_sb3:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    action = agent.q_net(obs_t).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1

            if terminated:
                crash_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

    env.close()

    rewards_arr = np.array(episode_rewards)
    return {
        "mean":        float(np.mean(rewards_arr)),
        "std":         float(np.std(rewards_arr)),
        "min":         float(np.min(rewards_arr)),
        "max":         float(np.max(rewards_arr)),
        "median":      float(np.median(rewards_arr)),
        "crash_rate":  crash_count / n_episodes,
        "mean_length": float(np.mean(episode_lengths)),
        "raw_rewards": episode_rewards,
    }


def evaluate_multi_seed_training(
    train_fn,
    train_seeds: list,
    eval_seeds: list,
    n_eval_episodes: int = 50,
    env_config: dict = None,
) -> dict:
    results = {}
    for train_seed in train_seeds:
        print(f"\n=== Entraînement avec seed={train_seed} ===")
        agent = train_fn(seed=train_seed)
        metrics = evaluate_agent(agent, env_config, n_eval_episodes, seeds=eval_seeds)
        results[train_seed] = metrics
        print(f"Seed {train_seed} | Mean: {metrics['mean']:.3f} ± {metrics['std']:.3f}")

    all_means = [r["mean"] for r in results.values()]
    print(f"\n=== Résultats agrégés sur {len(train_seeds)} seeds ===")
    print(f"Mean of means : {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
    return results


def print_eval_table(results: dict, model_name: str = "DQN"):
    print(f"\n{'='*55}")
    print(f"  Évaluation : {model_name}")
    print(f"{'='*55}")
    print(f"  {'Seed':>8} | {'Mean':>8} | {'Std':>7} | {'Crash%':>7}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
    for seed, m in results.items():
        print(f"  {seed:>8} | {m['mean']:>8.3f} | {m['std']:>7.3f} | "
              f"{m['crash_rate']*100:>6.1f}%")
    all_means = [m["mean"] for m in results.values()]
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
    print(f"  {'AVG':>8} | {np.mean(all_means):>8.3f} | "
          f"{np.std(all_means):>7.3f} |")
    print(f"{'='*55}\n")


def plot_eval_comparison(results_dqn: dict, results_sb3: dict,
                         save_path="eval_comparison.png"):
    import matplotlib.pyplot as plt

    dqn_rewards = [r for m in results_dqn.values() for r in m["raw_rewards"]]
    sb3_rewards = [r for m in results_sb3.values() for r in m["raw_rewards"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([dqn_rewards, sb3_rewards], labels=["DQN (maison)", "SB3"])
    ax.set_ylabel("Reward par épisode")
    ax.set_title("Comparaison DQN vs Stable-Baselines3 — highway-v0")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Graphique sauvegardé : {save_path}")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()

    agent = DQNAgent(
        cfg=HighwayDQNConfig(),
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    )
    env.close()

    agent.load_checkpoint(r"checkpoints\dqn_highway_step30000.pt")

    EVAL_SEEDS = list(range(100, 150))
    metrics_dqn = evaluate_agent(agent, SHARED_CORE_CONFIG, n_episodes=50, seeds=EVAL_SEEDS)
    print_eval_table({"seed42": metrics_dqn}, model_name="Mon DQN")