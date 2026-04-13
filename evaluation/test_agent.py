import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from agents.random_agent import RandomAgent
from agents.dqn_sb3 import SB3DQNAgent
from agents.dqn_per import PERDQNAgent, HighwayPERConfig
from agents.dqn_custom import DQNAgent, HighwayDQNConfig

from tqdm import tqdm
import numpy as np
import imageio
import gymnasium as gym
import highway_env  # noqa: F401


AGENT_REGISTRY = {
    "random": {
        "agent_type": "random",
        "checkpoint": None,
    },
    "dqn_custom": {
        "agent_type": "dqn_custom",
        "checkpoint": "checkpoints/dqn_custom_20260413-082750/model_dqn_custom.pt",
    },
    "sb3": {
        "agent_type": "sb3",
        "checkpoint": "checkpoints/sb3_dqn/model_dqn_sb3.zip",
    },
    "dqn_double": {
        "agent_type": "dqn_custom",
        "checkpoint": "checkpoints/dqn_20260411-135652/20260413-063222_dqn_highway_final.pt",
        "double_dqn": True,
    },
    "dqn_per": {
        "agent_type": "dqn_per",
        "checkpoint": "checkpoints/per_dqn_20260411-191026/20260412-021940_per_dqn_final.pt",
    },
    "dqn_double_per": {
        "agent_type": "dqn_per",
        "checkpoint": "checkpoints/20260412-084516_per_double_dqn/20260412-084516_per_double_dqn_final.pt",
        "double_dqn": True,
    },
}


def _load_agent(entry: dict, env: gym.Env):
    agent_type = entry["agent_type"]
    checkpoint = entry.get("checkpoint")

    if agent_type == "random":
        return RandomAgent(action_space=env.action_space,
                           observation_space=env.observation_space,
                           epsilon=None)

    if agent_type == "dqn_custom":
        cfg = HighwayDQNConfig(double_dqn=entry.get("double_dqn", False))
        agent = DQNAgent(cfg, env.observation_space.shape, env.action_space.n)
        if checkpoint:
            agent.load_checkpoint(checkpoint)
        return agent

    if agent_type == "dqn_per":
        cfg = HighwayPERConfig(double_dqn=entry.get("double_dqn", False))
        agent = PERDQNAgent(
            cfg, env.observation_space.shape, env.action_space.n)
        if checkpoint:
            agent.load_checkpoint(checkpoint)
        return agent

    if agent_type == "sb3":
        return SB3DQNAgent(model_path=checkpoint, env=env, determistic=True)

    raise ValueError(f"Unknown agent_type: {agent_type}")


def run_episode(
    agent_name: str,
    render: bool = True,
    checkpoint: str = None,
    save_gif: bool = False,
    gif_path: str = "episode.gif",
) -> tuple[float, int]:
    entry = dict(AGENT_REGISTRY[agent_name])
    if checkpoint:
        entry["checkpoint"] = checkpoint

    render_mode = "rgb_array" if save_gif else ("human" if render else None)
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    frames = []

    if save_gif:
        cfg = dict(SHARED_CORE_CONFIG)
        cfg.update({"screen_width": 600, "screen_height": 400,
                    "scaling": 10, "centering_position": [0.3, 0.5]})

        original_auto_render = env.unwrapped._automatic_rendering

        def _capture():
            original_auto_render()
            frames.append(env.render())

        env.unwrapped._automatic_rendering = _capture
        env.unwrapped.configure(cfg)
    else:
        env.unwrapped.configure(SHARED_CORE_CONFIG)

    obs, _ = env.reset()
    agent = _load_agent(entry, env)

    done = truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        action = agent.act(obs, epsilon=0.0)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if save_gif:
            frames.append(env.render())

    if save_gif and frames:
        print(f"\nSaving GIF to {gif_path} ...")
        imageio.mimsave(gif_path, frames, fps=20, loop=0)

    env.close()
    return total_reward, steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single agent on Highway-env")
    parser.add_argument("--agent", type=str, default="random",
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Agent to evaluate.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes.")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering.")
    parser.add_argument("--save", action="store_true",
                        help="Save episode as a GIF.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override the default checkpoint path for this agent.")
    args = parser.parse_args()

    render = not args.no_render
    checkpoint = args.model_path or AGENT_REGISTRY[args.agent].get(
        "checkpoint")

    if checkpoint and not os.path.isabs(checkpoint):
        checkpoint = os.path.join(ROOT_DIR, checkpoint)
        
    if args.agent != "random":
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"Checkpoint not found: {checkpoint}")
            exit(1)

    print(f"Agent      : {args.agent}")
    print(f"Episodes   : {args.episodes}")
    print(f"Render     : {render}")
    if checkpoint:
        print(f"Checkpoint : {checkpoint}")

    rewards, steps_list = [], []
    pbar = tqdm(range(args.episodes), desc="Evaluation", unit="ep")
    for _ in pbar:
        r, s = run_episode(
            agent_name=args.agent,
            render=render,
            checkpoint=args.model_path,
            save_gif=args.save,
            gif_path=os.path.join(ROOT_DIR, "results", "videos", f"episode_{args.agent}.gif")
        )
        rewards.append(r)
        steps_list.append(s)
        pbar.set_postfix(last=f"{r:.2f}", steps=s,
                         avg=f"{np.mean(rewards):.2f}")

    if rewards:
        print(
            f"\nMean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Mean steps  : {np.mean(steps_list):.1f}")
