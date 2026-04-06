import numpy as np
import torch
import gymnasium as gym
import highway_env


def evaluate_agent(
        agent,
        env_config:dict,
        n_episodes:int = 50,
        seeds:list | None = None,
        deterministic:bool = True,
        is_sb3_agent:bool = False,

)-> dict:
    """
    Evaluate a reinforcement learning agent on a specified environment.

    Args:
        agent: The reinforcement learning agent to be evaluated.
        env_config (dict): Configuration for the environment, including the environment name and any necessary parameters.
        n_episodes (int, optional): The number of episodes to run for evaluation. Defaults to 50.
        seeds (list | None, optional): A list of seeds for reproducibility. If None, no specific seeds will be used. Defaults to None.
        deterministic (bool, optional): Whether to use deterministic actions during evaluation. Defaults to True.
        is_sb3_agent (bool, optional): Whether the agent is a Stable Baselines 3 agent. Defaults to False.

    Returns:
        dict: A dictionary containing evaluation results, such as average reward and episode lengths.
    """
    # Implementation of the evaluation logic goes here
    pass


def evaluate_multi_seed_training(
    train_fn,
    train_seeds: list,
    eval_seeds: list,
    n_eval_episodes: int = 50,
    env_config: dict = None,
) -> dict:
    
    pass


def print_eval_table(results: dict, model_name: str = "DQN"):
    """
    Print a formatted evaluation table for the given results.

    Args:
        results (dict): A dictionary containing evaluation results, such as average reward and episode lengths.
        model_name (str, optional): The name of the model being evaluated. Defaults to "DQN".
    """

    pass

def plot_eval_comparison(results_dqn: dict, results_sb3: dict, save_path="eval_comparison.png"):
    pass