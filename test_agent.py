import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import argparse
import os
import numpy as np
import gymnasium as gym
from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG
from agents.random_agent import RandomAgent
# from agents.dqn_custom import DQNAgent
from agents.dqn_sb3 import SB3Agent
import highway_env
from tqdm import tqdm

def make_env():
    def _init():
        env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
        env.unwrapped.configure(SHARED_CORE_CONFIG)
        env.reset()
        return env
    return _init


def run_episode(agent_type="random", render=True, model_path=None):
    render_mode = "human" if render else None
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)

    obs, info = env.reset()

    if agent_type == "random":
        agent = RandomAgent(env.action_space)
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

# Need to rework, i think the parallelization is not working as expected when choosing the actions
# def run_parallel_episodes(agent_type="random", num_episodes=50, num_envs=4, model_path=None):
#     envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
#     obs, info = envs.reset()

#     if agent_type == "random":
#         agent = RandomAgent(envs.action_space)
#     elif agent_type == "sb3":
#         agent = SB3Agent(model_path=model_path, action_space=envs.action_space, determistic=True)
#     else:
#         raise ValueError(f"Agent inconnu : {agent_type}")

#     completed_episodes = 0
#     rewards_history = []
#     steps_history = []

#     current_rewards = np.zeros(num_envs)
#     current_steps = np.zeros(num_envs)
#     action_counts = np.zeros(envs.single_action_space.n) # Pour compter les actions [0, 1, 2, 3, 4]
#     pbar = tqdm(total=num_episodes, desc="Évaluation parallèle")

#     while completed_episodes < num_episodes:
#         actions = agent.act(obs)
        
#         for a in actions:
#             action_counts[int(a)] += 1
            
            
#         obs, reward, terminated, truncated, info = envs.step(actions)

#         current_rewards += reward
#         current_steps += 1

#         done = terminated | truncated
#         for i in range(num_envs):
#             if done[i]:
#                 if completed_episodes < num_episodes:
#                     rewards_history.append(current_rewards[i])
#                     steps_history.append(current_steps[i])
#                     completed_episodes += 1
#                     pbar.update(1) # Mise à jour de la barre ici
                    
#                 current_rewards[i] = 0
#                 current_steps[i] = 0
#                 completed_episodes += 1

#                 if completed_episodes % 10 == 0 or completed_episodes == num_episodes:
#                     print(
#                         f"Épisodes terminés : {completed_episodes}/{num_episodes}")

#                 if completed_episodes >= num_episodes:
#                     break

#     envs.close()
#     pbar.close()
    
#     # Affichage du bilan des actions
#     print("\n--- Analyse des décisions de l'agent ---")
#     actions_names = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
#     total_actions = np.sum(action_counts)
#     for i, count in enumerate(action_counts):
#         percentage = (count / total_actions) * 100
#         print(f"{actions_names[i]:12}: {int(count)} fois ({percentage:.1f}%)")
#     return rewards_history[:num_episodes], steps_history[:num_episodes]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluation des agents sur Highway-env")
    parser.add_argument("--agent", type=str,
                        default="random", choices=["random", "sb3"])
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    # parser.add_argument("--num-envs", type=int, default=1,
    #                     help="Nombre d'environnements parallèles")

    args = parser.parse_args()
    render = not args.no_render

    print(f"Lancement de {args.episodes} épisode(s) avec l'agent '{args.agent}'...")

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

    if args.episodes > 0:
        print("\n--- Statistiques Globales ---")
        print(
            f"Récompense moyenne : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Durée moyenne : {np.mean(steps):.1f} étapes")
