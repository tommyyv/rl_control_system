# python libraries and modules
import os
import pandas as pd
import matplotlib.pyplot as plt

# user-defined modules
from config import (
    RL_ALGORITHM_TYPE,
)


def dataframe_plot_results(dataframe_baseline: pd.DataFrame,
                           dataframe_hybrid: pd.DataFrame,
                           results_path: str) -> None:

    """Plots the CSV result files.

    Args:
        dataframe_baseline (pd.DataFrame): baseline DataFrame object
        dataframe_hybrid (pd.DataFrame): hybrid DataFrame object
        results_path (str): filepath to the results directory

    Returns:
        None.

    Raises:
        None.
    """
    episode_to_plot = 0

    dataframe_baseline_episode = dataframe_baseline[dataframe_baseline['episode']
                                                    == episode_to_plot]

    dataframe_hybrid_episode = dataframe_hybrid[dataframe_hybrid['episode'] ==
                                                episode_to_plot]

    # --- subplot 1: current rat vs setpoint ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(
        14, 10), sharex=False)

    axes1[0].plot(dataframe_baseline_episode['time'], dataframe_baseline_episode['current_return_air_temp'], label='Baseline Return Temp', color='blue', alpha=0.7)

    axes1[0].plot(dataframe_hybrid_episode['time'], dataframe_hybrid_episode['current_return_air_temp'], label=f'{RL_ALGORITHM_TYPE} Hybrid Return Temp', color='red', alpha=0.7)

    axes1[0].plot(dataframe_baseline_episode['time'], dataframe_baseline_episode['current_setpoint'], 'k--', label='Setpoint')  # Setpoint line

    axes1[0].set_xlabel('Time (sec)')
    axes1[0].set_ylabel('Return Air Temperature')
    axes1[0].set_title('Current RAT vs. Setpoint')
    axes1[0].legend()
    axes1[0].grid(True)

    # --- Subplot 2: Control Signal Comparison ---
    axes1[1].plot(dataframe_baseline_episode['time'], dataframe_baseline_episode['previous_sat_control_signal'], label='Baseline PID SAT', color='blue', alpha=0.7)
    axes1[1].plot(dataframe_hybrid_episode['time'], dataframe_hybrid_episode['previous_sat_control_signal'], label=f'{RL_ALGORITHM_TYPE} Hybrid SAT', color='red', alpha=0.7)

    axes1[1].set_xlabel('Time (sec)')
    axes1[1].set_ylabel('SAT Control Signal')
    axes1[1].set_title('Control Signal per Episode')
    axes1[1].legend()
    axes1[1].grid(True)

    plt.savefig(os.path.join(results_path, 'rat_vs_sp_and_control_sig.png'))

    # --- Total Reward Per Episode Comparison ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    baseline_rewards_total = dataframe_baseline.groupby('episode')[
        'reward'].sum()
    hybrid_rewards_total = dataframe_hybrid.groupby('episode')[
        'reward'].sum()

    ax2.plot(baseline_rewards_total.index, baseline_rewards_total.values,
             marker='o', linestyle='-', label='Baseline Total Reward')

    ax2.plot(hybrid_rewards_total.index, hybrid_rewards_total.values,
             marker='x', linestyle='--', label=f'{RL_ALGORITHM_TYPE} Hybrid Total Reward')

    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Total Reward per Episode')
    ax2.set_title(
        'Total Reward per Episode: Baseline PID vs. Hybrid (PID + RL)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(results_path, 'total_reward.png'))
