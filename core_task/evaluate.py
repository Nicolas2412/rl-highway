import numpy as np
import torch
import gymnasium as gym
import highway_env
from typing import Optional
from ..shared_core_config import SHARED_CORE_CONFIG


def evaluate_agent(
    agent,                          # HighwayDQNAgent, politique SB3 ou random
    env_config: dict,
    n_episodes: int = 50,
    seeds: Optional[list] = None,
    deterministic: bool = True,     # Pas d'exploration pendant l'éval
    is_sb3: bool = False,
) -> dict:
    """
    Évalue un agent sur N épisodes avec seeds fixes.

    Retourne un dict avec mean, std, min, max, et les rewards bruts.
    """
    if seeds is None:
        seeds = list(range(n_episodes))

    assert len(seeds) == n_episodes, "Un seed par épisode"

    env = gym.make("highway-v0")
    env.configure(env_config)

    episode_rewards = []
    episode_lengths = []
    crash_count = 0

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=seeds[ep_idx])
        total_reward = 0
        length = 0
        done = False

        while not done:
            if is_sb3:
                action, _ = agent.predict(obs, deterministic=deterministic)
            else:
                # DQN maison : greedy pur (epsilon=0)
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
    """
    Entraîne sur plusieurs seeds d'entraînement et évalue chaque agent.
    Permet de mesurer la variance de l'algorithme (pas seulement de l'évaluation).
    """
    from copy import deepcopy

    results = {}
    for train_seed in train_seeds:
        print(f"\n=== Entraînement avec seed={train_seed} ===")
        agent = train_fn(seed=train_seed)

        metrics = evaluate_agent(
            agent, env_config, n_eval_episodes, seeds=eval_seeds
        )
        results[train_seed] = metrics
        print(f"Seed {train_seed} | Mean: {metrics['mean']:.3f} ± {metrics['std']:.3f}")

    # Aggrégation inter-seeds
    all_means = [r["mean"] for r in results.values()]
    print(f"\n=== Résultats agrégés sur {len(train_seeds)} seeds ===")
    print(f"Mean of means : {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
    return results


def print_eval_table(results: dict, model_name: str = "DQN"):
    """Affiche un tableau d'évaluation style rapport."""
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


def plot_eval_comparison(results_dqn: dict, results_sb3: dict, save_path="eval_comparison.png"):
    """Boxplot comparant DQN maison vs SB3."""
    import matplotlib.pyplot as plt

    dqn_rewards = []
    for m in results_dqn.values():
        dqn_rewards.extend(m["raw_rewards"])

    sb3_rewards = []
    for m in results_sb3.values():
        sb3_rewards.extend(m["raw_rewards"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([dqn_rewards, sb3_rewards], labels=["DQN (maison)", "SB3"])
    ax.set_ylabel("Reward par épisode")
    ax.set_title("Comparaison DQN vs Stable-Baselines3 — highway-v0")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Graphique sauvegardé : {save_path}")


# ─── Usage ────────────────────────────────────────────────────────────────────


EVAL_SEEDS = list(range(100, 150))  # 50 seeds fixes identiques pour tous les modèles

metrics_dqn = evaluate_agent(my_agent, SHARED_CORE_CONFIG, n_episodes=50,
                              seeds=EVAL_SEEDS, deterministic=True)
print_eval_table({"seed42": metrics_dqn}, model_name="Mon DQN")