"""
Recherche d'hyperparamètres DQN avec Optuna.

Usage :
    python -m core_task.hparam.hparam_search --n-trials 30 --fresh   # repart d'une étude vierge
"""

import argparse
import json
import os
import random
import time

import numpy as np
import optuna
import torch
import gymnasium as gym
import highway_env  # noqa: F401
from tqdm import tqdm

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.dqn_custom import DQNAgent, HighwayDQNConfig

TRIAL_STEPS = 40_000
RESULTS_DIR = "hparam_results"
DB_PATH     = os.path.join(RESULTS_DIR, "optuna_study.db")
STUDY_NAME  = "highway_dqn"


def make_env(seed: int) -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


def evaluate_config(cfg: HighwayDQNConfig) -> float:
    """Entraîne un agent avec cfg sur TRIAL_STEPS steps, retourne R̄₂₀."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg.seed)
    obs, _ = env.reset(seed=cfg.seed)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent     = DQNAgent(cfg, obs_shape, n_actions)

    episode_rewards: list[float] = []
    ep_reward = 0.0

    for step in range(TRIAL_STEPS):
        agent.global_step = step
        action     = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        agent.buffer.push(obs, action, reward, obs, float(terminated))
        ep_reward += reward

        if terminated or truncated:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            agent.update()

        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()

    env.close()
    return float(np.mean(episode_rewards[-20:])) if episode_rewards else -999.0


def objective(trial: optuna.Trial) -> float:
    cfg = HighwayDQNConfig(
        seed             = 42,
        learning_rate    = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        gamma            = trial.suggest_float("gamma", 0.80, 0.99),
        batch_size       = trial.suggest_categorical("batch_size", [32, 64, 128]),
        buffer_capacity  = trial.suggest_categorical("buffer_cap", [10_000, 20_000, 30_000]),
        epsilon_decay_steps = trial.suggest_int("eps_decay", 50_000, 150_000, step=25_000),
        target_update_frequency = trial.suggest_int("target_upd", 25, 200, step=25),
        hidden_dims      = [trial.suggest_categorical("hidden", [128, 256, 512])]
                           * trial.suggest_int("n_layers", 1, 3),
        double_dqn       = trial.suggest_categorical("double_dqn", [True, False]),
        total_timesteps  = TRIAL_STEPS,
        checkpoint_dir   = os.path.join(RESULTS_DIR, f"trial_{trial.number}"),
        checkpoint_frequency = TRIAL_STEPS + 1,
    )
    return evaluate_config(cfg)


def main(n_trials: int, fresh: bool):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --fresh : supprime l'ancienne étude pour repartir proprement
    if fresh and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Ancienne étude supprimée ({DB_PATH})")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name    = STUDY_NAME,
        direction     = "maximize",
        storage       = f"sqlite:///{DB_PATH}",
        load_if_exists= not fresh,
        pruner        = optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler       = optuna.samplers.TPESampler(seed=42),
    )

    print(f"Démarrage : {n_trials} trials × {TRIAL_STEPS} steps")
    t0 = time.perf_counter()

    # Barre de progression manuelle — fiable quelle que soit la version d'Optuna
    with tqdm(total=n_trials, unit="trial", colour="cyan") as pbar:
        def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            best = study.best_value if study.best_trial else float("nan")
            pbar.set_postfix({"trial": trial.number, "val": f"{trial.value:.3f}", "best": f"{best:.3f}"})
            pbar.update(1)

        study.optimize(
            objective,
            n_trials       = n_trials,
            callbacks      = [callback],
            gc_after_trial = True,
            catch          = (Exception,),   # log les erreurs sans crasher toute la recherche
        )

    elapsed = time.perf_counter() - t0
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nTerminé en {elapsed/60:.1f} min — {len(completed)}/{n_trials} trials complétés")

    if not completed:
        print("Aucun trial complété — vérifier les erreurs ci-dessus.")
        return

    best = study.best_trial
    print(f"Meilleur trial #{best.number} — R̄ = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:20s} = {v}")

    result = {"best_value": best.value, "best_params": best.params}
    with open(os.path.join(RESULTS_DIR, "best_hparams.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--fresh", action="store_true",
                        help="Repart d'une étude vierge (supprime l'ancienne DB)")
    args = parser.parse_args()
    main(args.n_trials, args.fresh)