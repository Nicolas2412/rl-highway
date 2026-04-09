"""
Optimisation des hyperparamètres DQN avec Optuna (TPE + MedianPruner).

Stratégie :
  - Chaque trial entraîne l'agent sur TRIAL_STEPS timesteps (run court).
  - La métrique est la récompense moyenne sur les 20 derniers épisodes.
  - Le pruner élimine les trials sous-performants à mi-chemin.
  - Les meilleurs hyperparamètres sont sauvegardés dans best_hparams.json.

Usage :
    python hparam_search.py --n-trials 30 --n-envs 4
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

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.dqn_custom import DQNAgent, HighwayDQNConfig

TRIAL_STEPS  = 40_000   # steps par trial (compromis vitesse / signal)
NUM_ENVS     = 4        # envs parallèles pendant la recherche
RESULTS_DIR  = "hparam_results"


def make_env(seed: int):
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset(seed=seed)
        return env
    return _init


def run_trial(cfg: HighwayDQNConfig, num_envs: int, trial: optuna.Trial) -> float:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg.seed + i) for i in range(num_envs)]
    )
    obs, _ = envs.reset(seed=cfg.seed)

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    agent     = DQNAgent(cfg, obs_shape, n_actions)

    episode_rewards: list[float] = []
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)
    losses: list[float] = []

    prune_check_interval = TRIAL_STEPS // 4

    for step in range(TRIAL_STEPS):
        agent.global_step = step
        actions = agent.select_actions_batch(obs)
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)

        for i in range(num_envs):
            agent.buffer.push(obs[i], actions[i], rewards[i],
                              next_obs[i], float(terminated[i]))

        obs = next_obs
        current_rewards += rewards
        current_lengths += 1

        for i in range(num_envs):
            if (terminated | truncated)[i]:
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0.0
                current_lengths[i] = 0

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()

        # Rapport intermédiaire pour le pruner
        if step % prune_check_interval == 0 and step > 0 and episode_rewards:
            intermediate = float(np.mean(episode_rewards[-20:]))
            trial.report(intermediate, step)
            if trial.should_prune():
                envs.close()
                raise optuna.TrialPruned()

    envs.close()
    if not episode_rewards:
        return -999.0
    return float(np.mean(episode_rewards[-20:]))


def objective(trial: optuna.Trial) -> float:
    lr             = trial.suggest_float("learning_rate",   1e-4, 1e-3, log=True)
    gamma          = trial.suggest_float("gamma",           0.8,  0.99)
    batch_size     = trial.suggest_categorical("batch_size", [32, 64, 128])
    buffer_cap     = trial.suggest_categorical("buffer_capacity", [10_000, 15_000, 30_000])
    eps_decay      = trial.suggest_int("epsilon_decay_steps", 50_000, 150_000, step=25_000)
    target_update  = trial.suggest_int("target_update_frequency", 25, 200, step=25)
    n_layers       = trial.suggest_int("n_layers", 1, 3)
    hidden_size    = trial.suggest_categorical("hidden_size", [128, 256, 512])
    double_dqn     = trial.suggest_categorical("double_dqn", [True, False])

    hidden_dims = [hidden_size] * n_layers

    cfg = HighwayDQNConfig(
        seed=42,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_cap,
        epsilon_decay_steps=eps_decay,
        target_update_frequency=target_update,
        hidden_dims=hidden_dims,
        double_dqn=double_dqn,
        total_timesteps=TRIAL_STEPS,
        checkpoint_dir=os.path.join(RESULTS_DIR, f"trial_{trial.number}"),
        checkpoint_frequency=TRIAL_STEPS + 1,   # pas de checkpoint pendant la recherche
    )

    return run_trial(cfg, num_envs=NUM_ENVS, trial=trial)


def main(n_trials: int, n_envs: int):
    global NUM_ENVS
    NUM_ENVS = n_envs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    storage = f"sqlite:///{RESULTS_DIR}/optuna_study.db"
    study = optuna.create_study(
        study_name="highway_dqn",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"Démarrage de la recherche : {n_trials} trials, {TRIAL_STEPS} steps/trial")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nRecherche terminée en {(time.perf_counter()-t0)/60:.1f} min")

    best = study.best_trial
    print(f"\nMeilleur trial #{best.number} — R̄ = {best.value:.4f}")
    print("Hyperparamètres :")
    for k, v in best.params.items():
        print(f"  {k:30s} = {v}")

    out = {
        "best_value": best.value,
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
    }
    out_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSauvegardé dans {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-envs",   type=int, default=4)
    args = parser.parse_args()
    main(args.n_trials, args.n_envs)