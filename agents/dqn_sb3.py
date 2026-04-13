from agents.base_agent import BaseAgent
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Custom callback to extract all available informations
class HighwayMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.collision_count = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "speed" in info:
                self.logger.record("env/speed", info["speed"])

            if "rewards" in info:
                for key, val in info["rewards"].items():
                    self.logger.record(f"env/reward_{key}", val)

        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                if self.locals["infos"][i].get("crashed", False):
                    self.collision_count += 1

                self.logger.record("env/collision_rate",
                                self.collision_count / self.episode_count)
        return True
    
class SB3DQNAgent(BaseAgent):
    def __init__(self, cfg=None, env=None, model_path=None, tensorboard_log="results/logs/dqn_sb3/", **kwargs):
        self.cfg = cfg
        if model_path is not None:
            self.model = DQN.load(model_path, env=env)
        elif cfg is not None and env is not None:
            #Converting the custom config to the sb3 format
            sb3_params = {
                "learning_rate": cfg.learning_rate,
                "buffer_size":   cfg.buffer_capacity,
                "learning_starts": cfg.learning_starts,
                "batch_size":    cfg.batch_size,
                "tau":           1.0,
                "gamma":         cfg.gamma,
                "train_freq":    cfg.train_frequency,
                "gradient_steps": 1,
                "target_update_interval": cfg.target_update_frequency,
                "exploration_fraction": cfg.epsilon_decay_steps / cfg.total_timesteps,
                "exploration_initial_eps": cfg.epsilon_start,
                "exploration_final_eps": cfg.epsilon_end,
                "policy_kwargs": dict(net_arch=cfg.hidden_dims),
                "tensorboard_log": tensorboard_log,
                "verbose": 1,
            }
            self.model = DQN("MlpPolicy", env, **sb3_params)
            
    def act(self, obs, epsilon=None):
        action, state = self.model.predict(obs, deterministic=True)
        return int(action)

    def update(self, obs, action, reward, terminated, next_obs):
        pass

    def train(self, env, total_timesteps=10_000, seed=None, log_dir=None, run_name=None):
        
        save_path = self.cfg.checkpoint_dir if self.cfg else "./logs/checkpoints/"
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=save_path,
            name_prefix="sb3_ckpt"
        )
        
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name or "SB3_Run",
            callback=[HighwayMetricsCallback(), checkpoint_callback],
            reset_num_timesteps=False
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        current_env = getattr(self.model, "env", None) if hasattr(self, "model") else None
        self.model = DQN.load(path, env=current_env)


