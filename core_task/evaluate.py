# evaluate.py
import numpy as np

def evaluate_agent(agent, env, num_episodes=50, seed=None):
    """Works for ANY agent that implements BaseAgent."""
    rewards, lengths, successes, crash_times = [], [], [], []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        done, truncated = False, False
        total_reward, steps, crashed = 0, 0, False

        while not (done or truncated):
            action = agent.act(obs, epsilon=0.0)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if info.get("crashed", False):
                crashed = True

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(0 if crashed else 1)
        if crashed:
            crash_times.append(steps)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes) * 100,
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "mean_crash_time": np.mean(crash_times) if crash_times else None,
        "raw_rewards": rewards,
        "raw_lengths": lengths,
    }


def evaluate_over_seeds(agent_cls, agent_kwargs, make_env_fn,
                        seeds, num_train_episodes=500, num_eval_episodes=50,
                        weights_path=None):
    all_rewards, all_success_rates, all_lengths = [], [], []

    for seed in seeds:
        env = make_env_fn()
        agent = agent_cls(
            action_space=env.action_space,
            observation_space=env.observation_space,
            **agent_kwargs
        )

        if agent.needs_training:
            print("------ Starting training ------")
            agent.train(env, num_episodes=num_train_episodes, seed=seed)
            if weights_path:
                agent.save(weights_path.format(seed=seed))
            
        print("------ Starting evaluation ------")
        results = evaluate_agent(agent, env, num_episodes=num_eval_episodes, seed=seed)
        all_rewards.extend(results["raw_rewards"])
        all_success_rates.append(results["success_rate"])
        all_lengths.extend(results["raw_lengths"]) 
        env.close()

    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "success_rate": np.mean(all_success_rates),
        "mean_length": np.mean(all_lengths),
    }