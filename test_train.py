import gymnasium as gym
import highway_env
import os
from agents.dqn_sb3 import SB3DQNAgent 
from agents.dqn_custom import DQNAgent, HighwayDQNConfig

from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG 

def test_training(agent_type:str,
                gamma:float=0.99,
                batch_size:int=16,
                buffer_capacity:int=15_000,
                update_target_every:int=100,
                hidden_size:int=128,
                epsilon_start:float=1,
                decrease_epsilon_factor:float=100,
                epsilon_min:float=0.05,
                learning_rate:float=5e-4,
                exploration_fraction:float=0.1,
                num_episodes:int=10,
                run_name:str='test'
                ):
    
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    
    # Dossiers pour les sorties
    
    base_results_dir = f"results"
    log_dir = os.path.join(base_results_dir, "logs", agent_type)
    model_dir = os.path.join(base_results_dir, "models", agent_type)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{agent_type}_test_model")

    obs, info = env.reset()
    
    # Initialisation de l'agent
    if agent_type == "dqn_sb3":
        agent = SB3DQNAgent(
            env=env,
            policy_kwargs=dict(net_arch=[hidden_size, hidden_size]) ,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_capacity,
            target_update_interval=update_target_every,
            exploration_final_eps=epsilon_min,
            exploration_fraction=exploration_fraction,
            learning_starts=batch_size,
            verbose=1
        )
    elif agent_type == "dqn_custom":
        config = HighwayDQNConfig(gamma=gamma,
                                batch_size=batch_size,
                                buffer_capacity=buffer_capacity,
                                target_update_frequency=update_target_every,
                                hidden_dims=[hidden_size],
                                epsilon_start=epsilon_start,
                                epsilon_end=epsilon_min,
                                total_timesteps=num_episodes * 30,  # Juste une estimation grossière
                                learning_rate=learning_rate,
                                
                                )
        agent = DQNAgent(cfg=config,
                         obs_shape=env.observation_space.shape, 
                         n_actions=env.action_space.n,
           
        )
    else:
        raise ValueError(f"Agent '{agent_type}' not implemented")

    # Entraînement
    # On teste sur un petit nombre d'épisodes pour vérifier que le flux fonctionne
    print("Début du test d'entraînement...")
    agent.train(
        env, 
        num_episodes=num_episodes, 
        run_name=run_name,
        log_dir=log_dir
    )

    # Sauvegarde
    agent.save(model_path)
    print(f"Modèle sauvegardé dans : {model_path}.zip")

    # Test de l'action
    obs, info = env.reset()
    action = agent.act(obs)
    print(f"Action choisie par l'agent : {action}")

    env.close()
    print("Test terminé avec succès.")

if __name__ == "__main__":
    for agent_type in ["dqn_custom", "dqn_sb3"]:
        test_training(agent_type=agent_type,
                    learning_rate=5e-4,
                    gamma=0.99,
                    batch_size=16,
                    update_target_every=100,
                    buffer_capacity=15_000,
                    epsilon_start=1,
                    epsilon_min=0.05,
                    hidden_size=128,
                    run_name="test",
                    num_episodes=10,
                    
                    # Pour SB3 : fraction du temps total (0.1 = 10%)
                    exploration_fraction=0.1,
                    
                    # Pour Custom : facteur exponentiel
                    decrease_epsilon_factor=100,
                    )