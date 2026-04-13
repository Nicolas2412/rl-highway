#python -m core_task.plot_metrics

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import highway_env

from shared_core_config import (
    SHARED_CORE_ENV_ID, 
    SHARED_CORE_CONFIG, 
    SHARED_SEED, 
)
from evaluate import evaluate_over_seeds
from agents.random_agent import RandomAgent
from agents.dqn_custom import DQNAgent 
from agents.dqn_sb3 import SB3DQNAgent

DQN_CUSTOM_PARAMS = {
    "gamma": 0.95,
    "batch_size": 32,
    "buffer_capacity": 15000,
    "update_target_every": 50,
    "epsilon_start": 1.0,
    "decrease_epsilon_factor": 200,
    "epsilon_min": 0.05,
    "learning_rate": 5e-4,
    "hidden_size": 256,
}

DQN_SB3_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 5e-4,
    "buffer_size": 15000,
    "learning_starts": 200,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.95,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 50,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.05,
}

SHARED_SEED = 42

SEEDS = [SHARED_SEED, 123, 777] 

def make_env():
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset(seed=SHARED_SEED)
    return env

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
        "cls": DQNAgent,
        "kwargs": DQN_CUSTOM_PARAMS,
        "train_episodes": 500,
        "weights_path": "weights/dqn_custom_seed{seed}.pth",
    },
    {
        "name": "DQN SB3",
        "cls": SB3DQNAgent,
        "kwargs": DQN_SB3_PARAMS,
        "train_episodes": 500,
        "weights_path": "weights/dqn_sb3_seed{seed}.zip",
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