#python -m core_task.plot_metrics

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import highway_env
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from core_task.evaluate import evaluate_over_seeds
from agents.random_agent import RandomAgent
from agents.dqn_custom import DQN_Custom

SEEDS = [42, 123, 777]

def make_env():
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return env

# ← the only place agent differences are declared
AGENT_REGISTRY = [
    {
        "name": "Random",
        "cls": RandomAgent,
        "kwargs": {},
        "train_episodes": 0,
        "weights_path": None,
    },
    {
        "name": "DQN Custom",
        "cls": DQN_Custom,
        "kwargs": {},
        "train_episodes": 500,
        "weights_path": "weights/dqn_seed{seed}.pth",
    },
]

def plot_results(all_results):
    names = [r["name"] for r in all_results]
    metrics = ["mean_reward", "success_rate", "mean_length"]
    titles = ["Mean Reward", "Success Rate (%)", "Mean Episode Length"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(axes, metrics, titles):
        values = [r["results"][metric] for r in all_results]
        stds = [r["results"].get("std_reward", 0) if metric == "mean_reward" else 0
                for r in all_results]
        ax.bar(names, values, yerr=stds, capsize=5)
        ax.set_title(title)
        ax.set_ylabel(title)
    plt.tight_layout()
    plt.savefig("results/comparison.png")
    plt.show()


if __name__ == "__main__":
    all_results = []
    for entry in AGENT_REGISTRY:
        print(f"\nEvaluating {entry['name']}...")
        results = evaluate_over_seeds(
            agent_cls=entry["cls"],
            agent_kwargs=entry["kwargs"],
            make_env_fn=make_env,
            seeds=SEEDS,
            num_train_episodes=entry["train_episodes"],
            weights_path=entry["weights_path"],
        )
        all_results.append({"name": entry["name"], "results": results})
        print(f"  → mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")

    plot_results(all_results)