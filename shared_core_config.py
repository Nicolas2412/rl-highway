SHARED_CORE_ENV_ID = "highway-v0"

SHARED_SEED = 42

SHARED_CORE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [20, 25, 30],
    },
    "lanes_count": 4,
    "vehicles_count": 45,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "collision_reward": -1.5,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.7,
    "lane_change_reward": -0.02,
    "reward_speed_range": [22, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}

# --- Agent Specific Hyperparameters ---

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
