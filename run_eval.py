from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.random_agent import RandomAgent
from agents.dqn_sb3 import SB3DQNAgent
from agents.dqn_per import PERDQNAgent, HighwayPERConfig
from agents.dqn_custom import DQNAgent, HighwayDQNConfig
import hashlib
import json
import os
import sys
import warnings
from dataclasses import fields

import numpy as np
import gymnasium as gym
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import highway_env  # noqa: F401


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

SEEDS = [9, 42, 67]
NUM_EPISODES = 50
FORCE = False

SUMMARY_PATH = os.path.join(SCRIPT_DIR, "results", "eval_summary.json")

EVAL_REGISTRY = [
    {
        "name":       "Random",
        "agent_type": "random",
        "checkpoint": None,
    },
    {
        "name":       "DQN Custom",
        "agent_type": "dqn_custom",
        "checkpoint": "checkpoints/dqn_custom_20260413-082750/model_dqn_custom.pt",
    },
    {
        "name":       "SB3 DQN",
        "agent_type": "sb3",
        "checkpoint": "checkpoints/sb3_dqn/model_dqn_sb3.zip",
    },
    {
        "name":       "DQN Double",
        "agent_type": "dqn_custom",
        "checkpoint": "checkpoints/dqn_20260411-135652/20260413-063222_dqn_highway_final.pt",
        "double_dqn": True,
    },
    {
        "name":       "DQN PER",
        "agent_type": "dqn_per",
        "checkpoint": "checkpoints/per_dqn_20260411-191026/20260412-021940_per_dqn_final.pt",
    },
    {
        "name":       "DQN Double+PER",
        "agent_type": "dqn_per",
        "checkpoint": "checkpoints/20260412-084516_per_double_dqn/20260412-084516_per_double_dqn_final.pt",
        "double_dqn": True,
    },
]


def _make_env() -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return env


def _get_params_from_registry(checkpoint_path: str, registry_file: str = "runs_registry.jsonl") -> dict:
    if not checkpoint_path:
        return {}

    reg_path = os.path.join(SCRIPT_DIR, registry_file)
    if not os.path.exists(reg_path):
        return {}

    folder_name = os.path.basename(os.path.dirname(checkpoint_path))
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("run_id") == folder_name:
                    return data.get("hyperparameters", {})
    except Exception as e:
        print(f"Registry read error: {e}")

    print("[!] No params found in registry for this run.")
    return {}


def _build_config(config_class, params_dict: dict):
    valid_keys = {f.name for f in fields(config_class)}
    filtered = {k: v for k, v in params_dict.items() if k in valid_keys}
    return config_class(**filtered)


def _load_agent(entry: dict, env: gym.Env):
    agent_type = entry["agent_type"]
    checkpoint = entry.get("checkpoint")

    reg_params = _get_params_from_registry(
        checkpoint, "checkpoints/runs_registry.jsonl")
    merged = {**entry, **reg_params}

    if agent_type == "random":
        return RandomAgent(action_space=env.action_space,
                           observation_space=env.observation_space,
                           **merged)

    if agent_type == "dqn_custom":
        cfg = _build_config(HighwayDQNConfig, merged)
        agent = DQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        if checkpoint:
            agent.load_checkpoint(checkpoint, show=False)
        return agent

    if agent_type == "dqn_per":
        cfg = _build_config(HighwayPERConfig, merged)
        agent = PERDQNAgent(
            cfg, env.observation_space.shape, env.action_space.n)
        if checkpoint:
            agent.load_checkpoint(checkpoint, show=False)
        return agent

    if agent_type == "sb3":
        cfg = _build_config(HighwayDQNConfig, merged)
        return SB3DQNAgent(model_path=checkpoint, env=env) if checkpoint else SB3DQNAgent(cfg=cfg, env=env)

    raise ValueError(f"Unknown agent_type: {agent_type}")


def _run_episode(agent, env: gym.Env, seed: int | None) -> dict:
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    total_reward, steps, crashed, speeds = 0.0, 0, False, []

    while not (done or truncated):
        action = agent.act(obs, epsilon=0.0)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("crashed", False):
            crashed = True
        if "speed" in info:
            speeds.append(info["speed"])

    return {
        "reward":     total_reward,
        "length":     steps,
        "crashed":    crashed,
        "mean_speed": float(np.mean(speeds)) if speeds else None,
    }


def evaluate_agent(entry: dict, seed: int, num_episodes: int, pbar: tqdm | None = None) -> dict:
    env = _make_env()
    agent = _load_agent(entry, env)
    rewards, lengths, crashed, speeds = [], [], [], []

    for ep_idx in range(num_episodes):
        r = _run_episode(agent, env, seed=seed if ep_idx == 0 else None)
        rewards.append(r["reward"])
        lengths.append(r["length"])
        crashed.append(r["crashed"])
        if r["mean_speed"] is not None:
            speeds.append(r["mean_speed"])
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                seed=seed,
                R=f"{np.mean(rewards):.2f}",
                ok=f"{(1 - np.mean(crashed)) * 100:.0f}%",
            )

    env.close()
    crash_lengths = [l for l, c in zip(lengths, crashed) if c]
    return {
        "seed":            seed,
        "rewards":         rewards,
        "lengths":         lengths,
        "crashed":         crashed,
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "success_rate":    float(1 - np.mean(crashed)),
        "mean_length":     float(np.mean(lengths)),
        "mean_crash_step": float(np.mean(crash_lengths)) if crash_lengths else None,
        "mean_speed":      float(np.mean(speeds)) if speeds else None,
    }


def _aggregate(per_seed: list[dict]) -> dict:
    all_rewards = [r for s in per_seed for r in s["rewards"]]
    all_lengths = [l for s in per_seed for l in s["lengths"]]
    all_crashed = [c for s in per_seed for c in s["crashed"]]
    all_speeds = [s["mean_speed"]
                  for s in per_seed if s["mean_speed"] is not None]
    crash_steps = [s["mean_crash_step"]
                   for s in per_seed if s["mean_crash_step"] is not None]
    return {
        "mean_reward":     float(np.mean(all_rewards)),
        "std_reward":      float(np.std(all_rewards)),
        "median_reward":   float(np.median(all_rewards)),
        "success_rate":    float(1 - np.mean(all_crashed)),
        "mean_length":     float(np.mean(all_lengths)),
        "std_length":      float(np.std(all_lengths)),
        "mean_crash_step": float(np.mean(crash_steps)) if crash_steps else None,
        "mean_speed":      float(np.mean(all_speeds)) if all_speeds else None,
        "raw_rewards":     all_rewards,
        "raw_lengths":     all_lengths,
        "raw_crashed":     all_crashed,
    }


SUMMARY_KEYS = [
    "name", "agent_type", "checkpoint_used", "mean_reward", "std_reward",
    "median_reward", "success_rate", "mean_length", "std_length",
    "mean_speed", "mean_crash_step", "seeds", "num_episodes", "per_seed",
]


def _save_summary(all_results: list[dict]) -> None:
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump([{k: r.get(k) for k in SUMMARY_KEYS}
                  for r in all_results], f, indent=2)


def _print_summary(all_results: list[dict]) -> None:
    def fmt(v):
        return f"{v:.1f}" if v is not None else "N/A"

    col_w = [20, 9, 9, 11, 10, 8, 9]
    headers = ["Agent", "Reward", "± std",
               "Success %", "Ep.len", "Speed", "Crash.Step"]
    header = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    sep = "-" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")

    for r in all_results:
        row = [
            r["name"],
            f"{r['mean_reward']:.3f}",
            f"{r['std_reward']:.3f}",
            f"{r['success_rate'] * 100:.1f}",
            f"{r['mean_length']:.1f}",
            fmt(r.get("mean_speed")),
            fmt(r.get("mean_crash_step")),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

        for s in r.get("per_seed", []):
            s_row = [
                f"  ├─ Seed {s['seed']}",
                f"{s['mean_reward']:.3f}",
                f"{s['std_reward']:.3f}",
                f"{s['success_rate'] * 100:.1f}",
                f"{s['mean_length']:.1f}",
                fmt(s.get("mean_speed")),
                fmt(s.get("mean_crash_step")),
            ]
            print("  ".join(v.ljust(w) for v, w in zip(s_row, col_w)))

    print(sep)


def main() -> None:
    print(
        f"Seeds: {SEEDS}  |  Episodes: {NUM_EPISODES}/seed  |  Force: {FORCE}\n")

    existing = []
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    all_results = []

    for entry in EVAL_REGISTRY:
        checkpoint = entry.get("checkpoint")
        checkpoint_name = (
            os.path.relpath(checkpoint, SCRIPT_DIR).replace("\\", "/")
            if checkpoint else "No Checkpoint"
        )

        cached = next(
            (r for r in existing
             if r["name"] == entry["name"]
             and r.get("checkpoint_used") == checkpoint_name
             and r.get("num_episodes", 0) >= NUM_EPISODES),
            None,
        )

        if cached and not FORCE:
            print(f"[SKIP] {entry['name']}  (checkpoint: {checkpoint_name})")
            all_results.append(cached)
            continue

        print(f"[RUN]  {entry['name']}")
        per_seed = []
        with tqdm(total=NUM_EPISODES * len(SEEDS), desc=entry["name"], unit="ep") as pbar:
            for seed in SEEDS:
                per_seed.append(evaluate_agent(
                    entry, seed=seed, num_episodes=NUM_EPISODES, pbar=pbar))

        result = _aggregate(per_seed)
        result.update({
            "name":            entry["name"],
            "agent_type":      entry["agent_type"],
            "checkpoint_used": checkpoint_name,
            "seeds":           SEEDS,
            "num_episodes":    NUM_EPISODES,
            "per_seed": [
                {k: s[k] for k in ("seed", "mean_reward", "std_reward",
                                   "success_rate", "mean_length", "mean_speed", "mean_crash_step")}
                for s in per_seed
            ],
        })

        print(f"       R={result['mean_reward']:.3f} ± {result['std_reward']:.3f}"
              f"  |  success={result['success_rate'] * 100:.1f}%")

        all_results.append(result)
        _save_summary(all_results)

    if not all_results:
        print("No results — check EVAL_REGISTRY.")
        return

    _print_summary(all_results)
    print(f"\nSummary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
