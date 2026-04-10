<<<<<<< HEAD
<<<<<<< HEAD
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
=======
=======
>>>>>>> 310dd4f8b4f2ea2d862cce6cc01b1f86ef6c2187
"""
Usage : 
 
python -m run --agent dqn_custom --checkpoint "checkpoints/dqn_highway_step70000.pt"

"""

<<<<<<< HEAD
>>>>>>> 310dd4f ([Docs] Improve log of dqn training)
=======
>>>>>>> 310dd4f8b4f2ea2d862cce6cc01b1f86ef6c2187

import argparse
import os
import numpy as np
import gymnasium as gym
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from agents.random_agent import RandomAgent
<<<<<<< HEAD
from agents.dqn_sb3 import SB3Agent
from agents.dqn_custom import DQNAgent
=======
from agents.dqn_custom import DQNAgent, HighwayDQNConfig
# from agents.sb3_agent import SB3Agent
>>>>>>> 310dd4f8b4f2ea2d862cce6cc01b1f86ef6c2187
import highway_env
from tqdm import tqdm


def make_env():
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset()
        return env
    return _init


<<<<<<< HEAD
<<<<<<< HEAD
def run_episode(agent_type="random", render=True, model_path=None):
=======
def run_episode(agent_type="random", render=True, checkpoint_to_load = None):
>>>>>>> 310dd4f ([Docs] Improve log of dqn training)
=======
def run_episode(agent_type="random", render=True, checkpoint_to_load = None):
>>>>>>> 310dd4f8b4f2ea2d862cce6cc01b1f86ef6c2187
    render_mode = "human" if render else None
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)

    obs, info = env.reset()

    if agent_type == "random":
        agent = RandomAgent(env.action_space)
    elif agent_type == "dqn_custom":
        agent = DQNAgent(cfg=HighwayDQNConfig(), obs_shape=env.observation_space.shape, n_actions=env.action_space.n)
        if checkpoint_to_load :
            agent.load_checkpoint(checkpoint_to_load)
    elif agent_type == "sb3":
        agent = SB3Agent(model_path=model_path, action_space=env.action_space, determistic=True)
    else:
        raise ValueError(f"Agent inconnu : {agent_type}")

    done = False
    truncated = False
    total_reward = 0
    step = 0

    while not (done or truncated):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        if render:
            env.render()

    env.close()
    return total_reward, step


def run_parallel_episodes(agent_type="random", num_episodes=50, num_envs=4):
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
    obs, info = envs.reset()

    if agent_type == "random":
        agent = RandomAgent(envs.action_space)
    elif agent_type == "dqn_custom":
        obs_shape = envs.observation_space.shape[1:]
        agent = DQNAgent(cfg=HighwayDQNConfig(), obs_shape=obs_shape, n_actions=envs.action_space.n)
        agent.load_checkpoint(r"checkpoints\dqn_highway_step110000.pt")
    elif agent_type == "sb3":
        raise NotImplementedError("Modèle SB3 non lié.")
    else:
        raise ValueError(f"Agent inconnu : {agent_type}")

    completed_episodes = 0
    rewards_history = []
    steps_history = []

    current_rewards = np.zeros(num_envs)
    current_steps = np.zeros(num_envs)

    while completed_episodes < num_episodes:
        actions = agent.act(obs)
        obs, reward, terminated, truncated, info = envs.step(actions)

        current_rewards += reward
        current_steps += 1

        done = terminated | truncated
        for i in range(num_envs):
            if done[i]:
                rewards_history.append(current_rewards[i])
                steps_history.append(current_steps[i])

                current_rewards[i] = 0
                current_steps[i] = 0
                completed_episodes += 1

                if completed_episodes % 10 == 0 or completed_episodes == num_episodes:
                    print(
                        f"Épisodes terminés : {completed_episodes}/{num_episodes}")

                if completed_episodes >= num_episodes:
                    break

    envs.close()
    return rewards_history[:num_episodes], steps_history[:num_episodes]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluation des agents sur Highway-env")
    parser.add_argument("--agent", type=str,
                        default="random", choices=["random","dqn_custom","sb3"],)
<<<<<<< HEAD
    parser.add_argument("--agent", type=str, default="random", choices=["random", "sb3", "dqn_custom"])
=======
>>>>>>> 310dd4f8b4f2ea2d862cce6cc01b1f86ef6c2187
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")

    parser.add_argument("--checkpoint", type = str, default = r"checkpoints\dqn_highway_step110000.pt")
    args = parser.parse_args()
    render = not args.no_render

    print(f"Lancement de {args.episodes} épisode(s) avec l'agent '{args.agent}'...")

<<<<<<< HEAD
    # model_path = f"results/models/{args.agent}/{args.name}.zip"
    model_path = f"results/models/test.zip"
    if not os.path.exists(model_path) and args.agent != "random":
        print(f"Modèle non trouvé : {model_path}")
        print("Assurez-vous que le modèle existe ou exécutez d'abord l'entraînement.")
        exit(1)
    
    # if render:
    rewards = []
    steps = []
    pbar = tqdm(range(args.episodes), desc="Évaluation", unit="épisode")
    for i in pbar:
        reward, step = run_episode(agent_type=args.agent, render=render, model_path=model_path)
        rewards.append(reward)
        
        pbar.set_postfix({
            "Last_Rew": f"{reward:.2f}",
            "Steps": step,
            "Avg_Rew": f"{np.mean(rewards):.2f}"
        })
        
        steps.append(step)
    # else:
    #     print(f"Mode parallèle activé sur {args.num_envs} processus.")
    #     rewards, steps = run_parallel_episodes(
    #         agent_type=args.agent,
    #         num_episodes=args.episodes,
    #         num_envs=args.num_envs,
    #         model_path=model_path
    #     )
=======
    if render:
        rewards = []
        steps = []
        for i in range(args.episodes):
            reward, step = run_episode(agent_type=args.agent, render=render, checkpoint_to_load=args.checkpoint)
            rewards.append(reward)
            steps.append(step)
            if args.episodes <= 10 or (i + 1) % 5 == 0:
                print(
                    f"Épisode {i+1}/{args.episodes} - Récompense : {reward:.2f} - Étapes : {step}")
    else:
        print(f"Mode parallèle activé sur {args.num_envs} processus.")
        rewards, steps = run_parallel_episodes(
            agent_type=args.agent,
            num_episodes=args.episodes,
            num_envs=args.num_envs
        )
>>>>>>> 310dd4f ([Docs] Improve log of dqn training)

    if args.episodes > 0:
        print("\n--- Statistiques Globales ---")
        print(
            f"Récompense moyenne : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Durée moyenne : {np.mean(steps):.1f} étapes")
