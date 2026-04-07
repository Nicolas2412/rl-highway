TRAIN_ENV_ID = "highway-v0"

TRAIN_CONFIG = {
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
    "vehicles_count": 12,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "collision_reward": -5.0,
    "right_lane_reward": 0.0,
    "high_speed_reward": 2.5,
    "lane_change_reward": 0.2,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}
