"""
Évaluation multi-agents, multi-seeds sur la config de test imposée.
"""

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import highway_env  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Paramètres — modifier ici
# ---------------------------------------------------------------------------

SEEDS = [9, 42, 67]     # Ne pas changer de préference ou recalculer tout si changement
NUM_EPISODES = 1        # épisodes par seed, si un test avec plus d'episodes et enregistré, ces valeurs seront utilisés
FORCE = False           # True = ignore les résultats déja sauvegardés et les remplace

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
        "checkpoint": "checkpoints/old-runs/20260410-112718_dqn_highway_final_episodic.pt",
    },
    # {
    #     "name":       "SB3 DQN",
    #     "agent_type": "sb3",
    #     "checkpoint": "checkpoints/sb3_dqn/sb3_final.zip",
    # },
    # {
    #     "name":       "DQN Double",
    #     "agent_type": "dqn_custom",
    #     "checkpoint": "checkpoints/dqn_20260411-135652/20260411-154821_dqn_highway_step10000",
    #     "double_dqn": True,
    # },
    # {
    #     "name":       "DQN PER",
    #     "agent_type": "dqn_per",
    #     "checkpoint": "checkpoints/per_dqn_20260411-191026/dqn_20260411-135652",
    # },
#     {
#         "name":       "DQN Double+PER",
#         "agent_type": "dqn_per",
#         "checkpoint": "checkpoints/dqn_per_double-incomplete/20260412-071853_per_dqn_step170000.pt",
#         "double_dqn": True,
#     },
]

# ---------------------------------------------------------------------------
# Environnement et agents
# ---------------------------------------------------------------------------


def _make_env() -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return env


def _load_agent(entry: dict, env: gym.Env):
    agent_type = entry["agent_type"]
    checkpoint = entry.get("checkpoint")

    if agent_type == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(action_space=env.action_space,
                        observation_space=env.observation_space)

    if agent_type == "dqn_custom":
        from agents.dqn_custom import DQNAgent, HighwayDQNConfig
        cfg = HighwayDQNConfig(double_dqn=entry.get("double_dqn", False))
        agent = DQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        agent.load_checkpoint(checkpoint)
        return agent

    if agent_type == "dqn_per":
        from agents.dqn_per import PERDQNAgent, HighwayPERConfig
        cfg = HighwayPERConfig(double_dqn=entry.get("double_dqn", False))
        agent = PERDQNAgent(
            cfg, env.observation_space.shape, env.action_space.n)
        agent.load_checkpoint(checkpoint)
        return agent

    if agent_type == "sb3":
        from agents.dqn_sb3 import SB3DQNAgent
        return SB3DQNAgent(model_path=checkpoint, env=env)

    raise ValueError(f"agent_type inconnu : {agent_type}")


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

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


def evaluate_agent(entry: dict, seed: int, num_episodes: int,
                pbar: tqdm | None = None) -> dict:
    """Évalue un agent sur num_episodes épisodes pour une seed donnée."""
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Seeds : {SEEDS}  |  Épisodes : {NUM_EPISODES}/seed  |  Force : {FORCE}\n")
    
    all_results = []
    
    existing_summary = []
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, 'r') as f:
                existing_summary = json.load(f)
        except:
            existing_summary = []

    all_results = []
    
    for entry in EVAL_REGISTRY:
        checkpoint_path = entry.get("checkpoint")
        if checkpoint_path:
            checkpoint_name = os.path.relpath(checkpoint_path, SCRIPT_DIR).replace('\\', '/')
        else:
            checkpoint_name = "No Checkpoint"
        
        # Chercher si cet agent avec ce checkpoint précis est déjà dans le résumé
        match = next((item for item in existing_summary
                    if item["name"] == entry["name"]
                    and item.get("checkpoint_used") == checkpoint_name
                    and item.get("num_episodes", 0) >= NUM_EPISODES), None)
        
        if match and not FORCE:
            print(
                f"[SKIP] {entry['name']} trouvé dans le résumé (CP: {checkpoint_name})")
            all_results.append(match)
            continue
        else:
            print(f"[RUN]   {entry['name']}")
            per_seed = []
            with tqdm(total=NUM_EPISODES * len(SEEDS), desc=entry["name"], unit="ep") as pbar:
                for seed in SEEDS:
                    seed_result = evaluate_agent(entry, seed=seed,
                                                num_episodes=NUM_EPISODES, pbar=pbar)
                    per_seed.append(seed_result)
                    pbar.set_postfix(seed=seed,
                                    R=f"{seed_result['mean_reward']:.2f}",
                                    ok=f"{seed_result['success_rate']*100:.0f}%")

            result = _aggregate(per_seed)
            
            clean_per_seed = [
                {
                    "seed": s["seed"],
                    "mean_reward": s["mean_reward"],
                    "std_reward": s["std_reward"],
                    "success_rate": s["success_rate"],
                    "mean_length": s["mean_length"],
                    "mean_speed": s["mean_speed"],
                    "mean_crash_step": s["mean_crash_step"]
                } for s in per_seed
            ]
            
            result.update({"per_seed": clean_per_seed, "seeds": SEEDS,
                        "num_episodes": NUM_EPISODES,
                        "name": entry["name"], "agent_type": entry["agent_type"],
                        "checkpoint_used": checkpoint_name})
            
        print(f"        R={result['mean_reward']:.3f} ± {result['std_reward']:.3f}"
                f"  |  success={result['success_rate']*100:.1f}%")

        all_results.append(result)

    if not all_results:
        print("Aucun résultat — vérifiez EVAL_REGISTRY.")
        return

    # Résumé console
    col_w = [20, 9, 9, 11, 10, 8, 9]  # Ajout de colonnes pour Speed et Crash
    header = "  ".join(h.ljust(w) for h, w in
                       zip(["Agent / Seed", "Reward", "± std", "Success %", "Ep.len", "Speed", "Crash.Step"], col_w))
    print(f"\n{'-'*len(header)}\n{header}\n{'-'*len(header)}")

    # Petite fonction pour éviter de faire planter l'affichage si la valeur est None
    def fmt_val(v): return f"{v:.1f}" if v is not None else "N/A"

    for r in all_results:
        # Affichage global de l'agent
        row = [r["name"], f"{r['mean_reward']:.3f}", f"{r['std_reward']:.3f}",
               f"{r['success_rate']*100:.1f}", f"{r['mean_length']:.1f}",
               fmt_val(r.get("mean_speed")), fmt_val(r.get("mean_crash_step"))]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

        # Affichage détaillé par seed
        if "per_seed" in r:
            for s in r["per_seed"]:
                seed_name = f"  ├─ Seed {s['seed']}"
                s_row = [seed_name, f"{s['mean_reward']:.3f}", f"{s['std_reward']:.3f}",
                         f"{s['success_rate']*100:.1f}", f"{s['mean_length']:.1f}",
                         fmt_val(s.get("mean_speed")), fmt_val(s.get("mean_crash_step"))]
                print("  ".join(v.ljust(w) for v, w in zip(s_row, col_w)))
    print('-'*len(header))

    # Sauvegarde résumé
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump([{k: r.get(k) for k in
                    ["name", "agent_type", "checkpoint_used", "mean_reward", "std_reward",
                    "median_reward", "success_rate", "mean_length", "std_length",
                     "mean_speed", "mean_crash_step", "seeds", "num_episodes", "per_seed"]}
                   for r in all_results], f, indent=2)
    print(f"\nRésumé → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
