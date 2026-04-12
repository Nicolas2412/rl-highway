import os
import sys
import gymnasium as gym
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from agents.dqn_sb3 import SB3DQNAgent 
from agents.dqn_custom import DQNAgent, HighwayDQNConfig
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG

# CONFIGURATION ISSUE D'OPTUNA
BEST_HPARAMS = {
    "lr": 0.0001432249371823026,
    "gamma": 0.8296389588638785,
    "batch_size": 64,
    "buffer_cap": 30000,
    "eps_decay": 150000,
    "target_upd": 50,
    "hidden_size": 256,
    "n_layers": 2,
    "double_dqn": False
}


def register_run(agent_type, cfg, status="done", save_path=""):
    registry_path = os.path.join(ROOT_DIR, "checkpoints/runs_registry.jsonl")

    run_entry = {
        "run_id": f"{agent_type}_{time.strftime('%Y%m%d-%H%M%S')}",
        "algorithm": agent_type,
        "status": status,
        "started_at": time.strftime("%Y%m%d-%H%M%S"),
        "hyperparameters": {
            "lr": cfg.learning_rate,
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "buffer_capacity": cfg.buffer_capacity,
            "epsilon_decay_steps": cfg.epsilon_decay_steps,
            "target_update_frequency": cfg.target_update_frequency,
            "hidden_dims": cfg.hidden_dims,
            "double_dqn": cfg.double_dqn
        },
        "final_checkpoint": save_path
    }

    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_entry) + "\n")
    print(f"Run enregistré dans le registre : {registry_path}")

def run_benchmark(agent_type: str, num_episodes: int = 1000):
    print(f"\n--- DÉMARRAGE ENTRAÎNEMENT : {agent_type} ---")

    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()

    cfg = HighwayDQNConfig(
        learning_rate=BEST_HPARAMS["lr"],
        gamma=BEST_HPARAMS["gamma"],
        batch_size=BEST_HPARAMS["batch_size"],
        buffer_capacity=BEST_HPARAMS["buffer_cap"],
        epsilon_decay_steps=BEST_HPARAMS["eps_decay"],
        target_update_frequency=BEST_HPARAMS["target_upd"],
        hidden_dims=[BEST_HPARAMS["hidden_size"]] * BEST_HPARAMS["n_layers"],
        double_dqn=BEST_HPARAMS["double_dqn"],
        total_timesteps=num_episodes * 30  # Estimation pour SB3
    )

    if agent_type == "dqn_custom":
        agent = DQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        ext = "pt"
    elif agent_type == "dqn_sb3":
        agent = SB3DQNAgent(cfg=cfg, env=env)
        ext = "zip"
    else:
        raise ValueError("Type d'agent inconnu")
    run_name = f"bench_{agent_type}"
    
    log_dir = os.path.join("results/logs", agent_type)
    os.makedirs(log_dir, exist_ok=True)
    agent.train(env, num_episodes=num_episodes, run_name=run_name, log_dir=log_dir)
    print(f"Logging to {log_dir}")
    
    os.makedirs("results/benchmarks", exist_ok=True)
    save_path = f"results/benchmarks/model_{agent_type}.{ext}"
    agent.save(save_path)
    print(f"Modèle {agent_type} sauvegardé dans: {save_path}")
    register_run(agent_type, cfg, status="done", save_path=save_path)
    env.close()


if __name__ == "__main__":
    for algorithm in ["dqn_sb3"]:
        run_benchmark(agent_type=algorithm, num_episodes=6666)
