import argparse
import os
import numpy as np
import gymnasium as gym
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from agents.random_agent import RandomAgent
from agents.dqn_custom import DQN_Custom
import highway_env



def make_env():
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset()
        return env
    return _init


def run_episode(agent_type="random", render=True):
    render_mode = "human" if render else None
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)

    obs, info = env.reset()

    if agent_type == "random":
        agent = RandomAgent(env.action_space)
    elif agent_type == "dqn_custom":
        agent = DQN_Custom(env.action_space, env.observation_space)
    elif agent_type == "sb3":
        raise NotImplementedError("Modèle SB3 non lié.")
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
        agent = DQN_Custom(envs.action_space, envs.observation_space)
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
                        default="random", choices=["random", "dqn_custom"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Nombre d'environnements parallèles")

    args = parser.parse_args()
    render = not args.no_render

    print(
        f"Lancement de {args.episodes} épisode(s) avec l'agent '{args.agent}'...")

    if render:
        rewards = []
        steps = []
        for i in range(args.episodes):
            reward, step = run_episode(agent_type=args.agent, render=render)
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

    if args.episodes > 1:
        print("\n--- Statistiques Globales ---")
        print(
            f"Récompense moyenne : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Durée moyenne : {np.mean(steps):.1f} étapes")
