"""
Évaluation multi-agents, multi-seeds sur la config de test imposée.
"""

from agents.dqn_custom import DQNAgent, HighwayDQNConfig
from agents.random_agent import RandomAgent
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import hashlib
import json
import os
import sys
import warnings
from dataclasses import fields

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import highway_env  # noqa: F401

from agents.dqn_per import PERDQNAgent, HighwayPERConfig
from agents.dqn_sb3 import SB3DQNAgent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Paramètres — modifier ici
# ---------------------------------------------------------------------------

SEEDS = [9, 42, 67]     # Ne pas changer de préference ou recalculer tout si changement
NUM_EPISODES = 50        # épisodes par seed, si un test avec plus d'episodes et enregistré, ces valeurs seront utilisés
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

# ---------------------------------------------------------------------------
# Environnement et agents
# ---------------------------------------------------------------------------


def _make_env() -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()
    return env


def _get_params_from_registry(checkpoint_path: str, registry_file: str = "runs_registry.jsonl") -> dict:
    """Cherche l'architecture du modèle dans le fichier de registre d'entraînement."""
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

        print(f"Erreur de lecture du registre : {e}")

    print("[!] Paramètres absents du registry run")

    return {}


def _build_config(config_class, params_dict: dict):
    """
    Reconstruit l'objet de configuration utilisé pour initialiser les agents
    """
    valid_keys = {f.name for f in fields(config_class)}
    filtered_params = {k: v for k, v in params_dict.items() if k in valid_keys}
    return config_class(**filtered_params)


def _load_agent(entry: dict, env: gym.Env):
    agent_type = entry["agent_type"]
    checkpoint = entry.get("checkpoint")

    reg_params = _get_params_from_registry(checkpoint, "checkpoints/runs_registry.jsonl")
    merged_params = {**entry, **reg_params}

    if agent_type == "random":
        return RandomAgent(action_space=env.action_space,
                        observation_space=env.observation_space,
                           **merged_params)

    elif agent_type == "dqn_custom":
        cfg = _build_config(HighwayDQNConfig, merged_params)
        agent = DQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        if checkpoint:
            agent.load_checkpoint(checkpoint,show=False)
        return agent

    elif agent_type == "dqn_per":
        cfg = _build_config(HighwayPERConfig, merged_params)
        agent = PERDQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        if checkpoint:
            agent.load_checkpoint(checkpoint, show=False)
        return agent

    elif agent_type == "sb3":
        cfg = _build_config(HighwayDQNConfig, merged_params)

        if checkpoint:
            agent = SB3DQNAgent(model_path=checkpoint, env=env)
        else:
            agent = SB3DQNAgent(cfg=cfg, env=env)

        return agent

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
            current_mean_r = float(np.mean(rewards))
            current_success = float(1 - np.mean(crashed))
            pbar.set_postfix(seed=seed,
                            R=f"{current_mean_r:.2f}",
                            ok=f"{current_success*100:.0f}%")

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

        os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump([{k: r.get(k) for k in
                        ["name", "agent_type", "checkpoint_used", "mean_reward", "std_reward",
                        "median_reward", "success_rate", "mean_length", "std_length",
                            "mean_speed", "mean_crash_step", "seeds", "num_episodes", "per_seed"]}
                    for r in all_results], f, indent=2)
            
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

    print(f"\nRésumé → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
