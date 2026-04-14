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
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from agents.dqn_custom import DQNAgent, HighwayDQNConfig

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

REGISTRY_PATH = os.path.join(ROOT_DIR, "checkpoints", "runs_registry.jsonl")


def register_run_start(run_id, agent_type, cfg, run_dir):
    """Enregistre le début du run avec le statut 'running'"""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

    run_entry = {
        "run_id": run_id,
        "algorithm": agent_type,
        "status": "running",
        "started_at": time.strftime("%Y%m%d-%H%M%S"),
        "ended_at": None,
        "checkpoint_dir": run_dir,
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
        "final_checkpoint": None
    }

    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_entry) + "\n")
        

def register_run_end(run_id, final_checkpoint_path):
    """Met à jour le registre pour passer le run en 'done'"""
    if not os.path.exists(REGISTRY_PATH):
        return

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        entry = json.loads(line)
        if entry["run_id"] == run_id:
            entry["status"] = "done"
            entry["ended_at"] = time.strftime("%Y%m%d-%H%M%S")
            entry["final_checkpoint"] = final_checkpoint_path
        updated_lines.append(json.dumps(entry) + "\n")

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)
    print(f"Registre mis à jour (Run terminé) : {REGISTRY_PATH}")
    
    
def run_benchmark(agent_type: str, total_timesteps: int = 10_000):

    # Creation of a unique run ID based on agent type and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{agent_type}_{timestamp}"

    run_dir = os.path.join(ROOT_DIR, "checkpoints", run_id)
    log_dir = os.path.join(run_dir, "logs")  # TensorBoard logs
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n--- DÉMARRAGE ENTRAÎNEMENT : {run_id} ---")
    print(f"Dossier du run : {run_dir}")

    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset()

    # Configuration
    cfg = HighwayDQNConfig(
        learning_rate=BEST_HPARAMS["lr"],
        gamma=BEST_HPARAMS["gamma"],
        batch_size=BEST_HPARAMS["batch_size"],
        buffer_capacity=BEST_HPARAMS["buffer_cap"],
        epsilon_decay_steps=BEST_HPARAMS["eps_decay"],
        target_update_frequency=BEST_HPARAMS["target_upd"],
        hidden_dims=[BEST_HPARAMS["hidden_size"]] * BEST_HPARAMS["n_layers"],
        double_dqn=BEST_HPARAMS["double_dqn"],
        total_timesteps=total_timesteps,
        checkpoint_dir=run_dir
    )

    register_run_start(run_id, agent_type, cfg, run_dir)
    
    if agent_type == "dqn_custom":
        agent = DQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        ext = "pt"
    elif agent_type == "dqn_sb3":
        agent = SB3DQNAgent(cfg=cfg, env=env, tensorboard_log=log_dir)
        ext = "zip"
    else:
        raise ValueError("Type d'agent inconnu")
    
    agent.train(env, total_timesteps=total_timesteps,
                run_name=run_id, log_dir=log_dir)

    final_model_path = os.path.join(run_dir, f"final_model.{ext}")
    agent.save(final_model_path)
    print(f"Modèle {agent_type} final sauvegardé dans : {final_model_path}")
    
    register_run_end(run_id, final_model_path)
    env.close()

if __name__ == "__main__":
    # "dqn_sb3", "dqn_custom"
    agents_type = ["dqn_custom", "dqn_sb3"]
    for algorithm in agents_type:
        run_benchmark(agent_type=algorithm, total_timesteps=1000)