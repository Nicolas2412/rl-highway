"""
Recherche d'hyperparamètres DQN avec Optuna.

Usage :
    python -m training.hparam.hparam_search --n-trials 20 --fresh
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
import highway_env 
from tqdm import tqdm

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.dqn_custom import DQNAgent, HighwayDQNConfig

TRIAL_STEPS      = 10_000
PRUNE_CHECK_FREQ = 1_000   # fréquence de rapport intermédiaire au pruner
RESULTS_DIR      = "hparam_results"
DB_PATH          = os.path.join(RESULTS_DIR, "optuna_study_test.db")
STUDY_NAME       = "highway_dqn"


# ---------------------------------------------------------------------------
# Environnement
# ---------------------------------------------------------------------------

def make_env(seed: int) -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


# ---------------------------------------------------------------------------
# Évaluation d'une configuration
# ---------------------------------------------------------------------------

def evaluate_config(
    cfg: HighwayDQNConfig,
    trial: optuna.Trial,
    pbar: tqdm,
) -> float:
    """
    Entraîne un agent sur TRIAL_STEPS steps et retourne la récompense moyenne
    sur les 20 derniers épisodes.

    Le pruner Optuna est alimenté toutes les PRUNE_CHECK_FREQ steps via
    trial.report() + trial.should_prune().

    Parameters
    ----------
    cfg   : configuration de l'agent
    trial : objet Optuna utilisé pour le reporting intermédiaire et le pruning
    pbar  : barre de progression partagée entre tous les trials
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg.seed)
    obs, _ = env.reset(seed=cfg.seed)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n


    cfg.checkpoint_dir       = os.path.join(RESULTS_DIR, f"trial_{trial.number}")
    cfg.checkpoint_frequency = TRIAL_STEPS + 1   # pas de checkpoint intermédiaire

    agent = DQNAgent(cfg, obs_shape, n_actions)

    episode_rewards: list[float] = []
    ep_reward = 0.0
    n_trials  = trial.study.sampler._n_startup_trials if hasattr(
        trial.study.sampler, "_n_startup_trials") else "?"

    for step in range(TRIAL_STEPS):

        agent.global_step = step

        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_reward += reward

        if terminated or truncated:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            agent.update()

        if step % cfg.target_update_frequency == 0:
            agent.sync_target_network()


        if step > 0 and step % PRUNE_CHECK_FREQ == 0 and episode_rewards:
            mean_r = float(np.mean(episode_rewards[-20:]))
            trial.report(mean_r, step=step)
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

        mean_r = np.mean(episode_rewards[-20:]) if episode_rewards else float("nan")
        pbar.set_description(
            f"Trial {trial.number} | "
            f"ε {agent.get_epsilon():.2f} | "
            f"R̄₂₀ {mean_r:.3f}"
        )
        pbar.update(1)

    env.close()
    return float(np.mean(episode_rewards[-20:])) if episode_rewards else -999.0


# ---------------------------------------------------------------------------
# Objectif Optuna
# ---------------------------------------------------------------------------

def make_objective(pbar: tqdm):
    """
    Retourne la fonction objectif qui sera appelée par study.optimize().
    """
    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden", [128, 256, 512])
        n_layers    = trial.suggest_int("n_layers", 1, 3)

        cfg = HighwayDQNConfig(
            seed                    = 42,
            learning_rate           = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            gamma                   = trial.suggest_float("gamma", 0.80, 0.99),
            batch_size              = trial.suggest_categorical("batch_size", [32, 64, 128]),
            buffer_capacity         = trial.suggest_categorical("buffer_cap", [10_000, 20_000, 30_000]),
            epsilon_decay_steps     = trial.suggest_int("eps_decay", 50_000, 150_000, step=25_000),
            target_update_frequency = trial.suggest_int("target_upd", 25, 200, step=25),
            hidden_dims             = [hidden_size] * n_layers,
            double_dqn              = trial.suggest_categorical("double_dqn", [True, False]),
            total_timesteps         = TRIAL_STEPS,
        )
        return evaluate_config(cfg, trial, pbar)

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_trials: int, fresh: bool) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if fresh and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Ancienne étude supprimée ({DB_PATH})")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name     = STUDY_NAME,
        direction      = "maximize",
        storage        = f"sqlite:///{DB_PATH}",
        load_if_exists = not fresh,
        pruner         = optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler        = optuna.samplers.TPESampler(seed=42),
    )

    total_steps = n_trials * TRIAL_STEPS
    print(f"Démarrage : {n_trials} trials × {TRIAL_STEPS} steps = {total_steps:,} steps au total")
    t0 = time.perf_counter()


    with tqdm(total=total_steps, unit="step", dynamic_ncols=True, colour="cyan") as pbar:
        study.optimize(
            make_objective(pbar),
            n_trials       = n_trials,
            gc_after_trial = True,
            catch          = (Exception,),
        )

    elapsed   = time.perf_counter() - t0
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(
        f"\nTerminé en {elapsed/60:.1f} min — "
        f"{len(completed)}/{n_trials} complétés, "
        f"{len(pruned)} élagués"
    )

    if not completed:
        print("Aucun trial complété — vérifier les erreurs ci-dessus.")
        return

    best = study.best_trial
    print(f"Meilleur trial #{best.number} — R̄ = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v}")

    result = {"best_value": best.value, "best_params": best.params}
    out_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Résultats sauvegardés → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Repart d'une étude vierge (supprime l'ancienne DB)",
    )
    args = parser.parse_args()
    main(args.n_trials, args.fresh)