# python libraries & modules
import os
from stable_baselines3.common.env_util import make_vec_env

# user-defined modules
from data_processing.dataset_loader import DatasetLoader
from data_processing.create_dataframe import create_dataframe
from data_processing.run_baseline_evaluation import run_baseline_evaluation
from data_processing.run_hybrid_evaluation import run_hybrid_evaluation
from data_processing.train_rl_agent import train_rl_agent
from data_processing.system_identifier import SystemIdentifier
from environment.create_temperature_env_instance import create_temperature_env_instance
from data_visualization.dataframe_plot_results import dataframe_plot_results
from data_visualization.error_metrics import error_metrics_file
from utils.save_results import save_results
from config import (
    DATASET_PATH,
    RL_N_STEPS,
    RL_LEARNING_RATE,
    RL_ALGORITHM_TYPE,
    RL_BATCH_SIZE,
    N_EVAL_EPISODES,
    RL_TOTAL_TIMESTEPS,
    RL_GAMMA,
    LOG_DIRECTORY,
    SAVE_PATH,
    RESULTS_DIRECTORY,
    N_ENVIRONMENT
)

# create directories if they dont exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)


def main():
    print('executing program....(takes time)....')
    # -- PRE-PROCESS DATA --
    # load data
    dataset_loader = DatasetLoader(DATASET_PATH)
    dataset = dataset_loader.load_data()

    # define & initialize dataframe w/ columns
    dataframe = create_dataframe(dataset)

    dataframe_target_col = 'Return Air Temperature'
    dataframe_data_cols_list = ['Supply Air Temperature',
                                'Return Air Temperature',
                                'Outdoor Air Temperature']

    single_row_data_col = dataframe[dataframe_data_cols_list].iloc[0].values

    setpoint_data = dataframe['Supply Air Temperature Setpoint'].values
    outdoor_air_temp_data = dataframe['Outdoor Air Temperature'].values
    supply_air_temp_data = dataframe['Supply Air Temperature'].values

    # -- SYSTEM IDENTIFIER & ENVIRONMENT --
    # define & initialize system identifier
    system_identifier = SystemIdentifier()

    system_identifier.train_model(dataframe,
                                  dataframe_target_col,
                                  dataframe_data_cols_list)

    # system identifier - predict next values
    predicted_next_state = system_identifier.predict_next_state(
        single_row_data_col)

    # define & initialize environment
    def system_env_instance():
        return create_temperature_env_instance(system_model=system_identifier,
                                               outdoor_air_temp_source=outdoor_air_temp_data,
                                               setpoint_source=setpoint_data,
                                               supply_air_temp_source=supply_air_temp_data)

    # execute baseline evalution - before RL algorithm
    dataframe_baseline_results = run_baseline_evaluation(system_env_instance,
                                                         num_episodes=N_EVAL_EPISODES)

    results_dir = save_results(RESULTS_DIRECTORY)
    dataframe_baseline_results.to_csv(os.path.join(results_dir,
                                                   'baseline_pid_results.csv'), index=True)

    # define and initialize RL agent - ppo
    train_vec_env = make_vec_env(system_env_instance,
                                 n_envs=N_ENVIRONMENT,
                                 seed=0)

    train_rl_agent(env=train_vec_env,
                   algorithm_type=RL_ALGORITHM_TYPE,
                   total_timesteps=RL_TOTAL_TIMESTEPS,
                   n_steps=RL_N_STEPS,
                   learning_rate=RL_LEARNING_RATE,
                   gamma=RL_GAMMA,
                   batch_size=RL_BATCH_SIZE,
                   n_eval_episodes=N_EVAL_EPISODES,
                   save_path=SAVE_PATH,
                   log_dir=LOG_DIRECTORY,
                   )

    train_vec_env.close()

    saved_trained_rl_model = os.path.join(SAVE_PATH, f'{RL_ALGORITHM_TYPE}_final_model')

    # evaluate RL agent performance - hybrid system (PID + RL)
    dataframe_hybrid_results = run_hybrid_evaluation(system_env_instance,
                                                     saved_trained_rl_model,
                                                     num_episodes=N_EVAL_EPISODES)

    dataframe_hybrid_results.to_csv(os.path.join(results_dir,
                                                 'hybrid_results.csv'), index=True)

    error_metrics_file(results_dir)

    # -- DATA ANALYSIS & VISUALIZATION --
    dataframe_plot_results(dataframe_baseline_results,
                           dataframe_hybrid_results,
                           results_dir)


if __name__ == '__main__':
    main()
