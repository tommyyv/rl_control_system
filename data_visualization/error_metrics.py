# python libraries and modules
import os
import numpy as np
import pandas as pd


def error_metrics_file(results_dir: str) -> None:
    """Generates an error metric text file.

    Args:
        results_dir (str): results directory filepath

    Returns:
        None.

    Raises:
        None.
    """
    dataframe_baseline_results_analyzed = pd.read_csv(f'{results_dir}/baseline_pid_results.csv')
    dataframe_hybrid_results_analyzed = pd.read_csv(f'{results_dir}/hybrid_results.csv')

    # find the avg reward for each dataframe by taking the sum and mean
    baseline_avg_reward = dataframe_baseline_results_analyzed.groupby('episode')[
        'reward'].sum().mean()
    hybrid_avg_reward = dataframe_hybrid_results_analyzed.groupby('episode')[
        'reward'].sum().mean()

    baseline_mae = dataframe_baseline_results_analyzed['return_air_temp_error'].abs(
    ).mean()

    baseline_rmse = np.sqrt(
        (dataframe_baseline_results_analyzed['return_air_temp_error']**2).mean())

    hybrid_mae = dataframe_hybrid_results_analyzed['return_air_temp_error'].abs(
    ).mean()

    hyrid_rmse = np.sqrt(
        (dataframe_hybrid_results_analyzed['return_air_temp_error']**2).mean())

    error_metrics_filepath = os.path.join(results_dir, 'error_metrics.txt')

    with open(error_metrics_filepath, 'w') as error_metrics_file:
        error_metrics_file.write(
            '--------- Error metrics ----------' + os.linesep)

        if hybrid_avg_reward > baseline_avg_reward:
            error_metrics_file.write("hybrid is more efficient" + os.linesep)
        else:
            error_metrics_file.write("baseline is more effieient" + os.linesep)

        error_metrics_file.write(
            f'Baseline MAE: {baseline_mae:.4f}' + os.linesep)
        error_metrics_file.write(f'Baseline RMSE: {baseline_rmse:.4f}' + os.linesep)
        error_metrics_file.write(f'Baseline Rewards Average: {baseline_avg_reward:.4f}' + os.linesep)
        error_metrics_file.write(f'Hybrid MAE: {hybrid_mae:.4f}' + os.linesep)
        error_metrics_file.write(f'Hybrid RMSE: {hyrid_rmse:.4f}' + os.linesep)
        error_metrics_file.write(f'Hybrid Rewards Average: {hybrid_avg_reward:.4f}' + os.linesep)
        error_metrics_file.write('----------------------' + os.linesep)
        error_metrics_file.close()
