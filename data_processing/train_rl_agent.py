# python libraries & modules
import os
from typing import Type
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# user-defined modules
from environment.temperature_env import TemperatureControlEnv


def train_rl_agent(
    env: Type[TemperatureControlEnv],
    algorithm_type: str,
    total_timesteps: int,
    learning_rate: float,
    gamma: float,
    n_steps: int,
    batch_size: int,
    save_path: str,
    log_dir: str,
    n_eval_episodes: int,
    policy_kwargs: dict = None
) -> sb3.common.base_class.BaseAlgorithm:
    """
    Trains the specified Reinforcement Learning agent using Stable Baselines3.

    Args:
        env (TemperatureControlEnv): The Gymnasium environment (or VecEnv) for training. This environment
             should be created using make_vec_env and potentially wrapped with Monitor.
        algorithm_type (str): The name of the RL algorithm (e.g., "PPO", "SAC").
        total_timesteps (int): The total number of timesteps for the agent to learn.
        learning_rate (float): The learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        n_steps (int): The number of steps to collect before each policy update (PPO specific).
        batch_size (int): The batch size for policy updates.
        save_path (str): Directory to save the best and final trained models.
        log_dir (str): Directory for TensorBoard logs and monitor CSV files.
        n_eval_episodes (int): Number of episodes to run during evaluation callback.
        policy_kwargs (dict, optional): optional keywords

    Returns:
        (stable_baselines3.common.base_class.BaseAlgorithm): The trained RL model instance.

    Raises:
        None
    """
    print(f"\n--- Training {algorithm_type} Agent ---")

    if algorithm_type == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs
        )
    else:
        raise ValueError(f'Unsupported RL algorithm type: {algorithm_type}. Only PPO is implemented in this function.')

    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_path,
        log_path=log_dir,
        eval_freq=max(total_timesteps // 10, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(os.path.join(save_path, f'{algorithm_type}_final_model'))
    print(f"{algorithm_type} agent training complete. Final model saved to {os.path.join(save_path, f'{algorithm_type}_final_model')}")

    return model
