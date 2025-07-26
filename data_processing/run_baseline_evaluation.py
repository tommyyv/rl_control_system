# python libraries & modules
from typing import Type
import pandas as pd
import numpy as np

# user-defined modules
from environment.temperature_env import TemperatureControlEnv
from config import (
    SIMULATED_DURATION_STEPS,
    DELTA_T_IN_SEC
)


def run_baseline_evaluation(env_instance: Type[TemperatureControlEnv], num_episodes: int) -> pd.DataFrame:
    """Executes the baseline evalation using a PID controller only.

    Args:
        env_instance (TemperatureControlEnv): the environment class instance for
            the trained model to run in
        num_episodes (int): number of episodes the evalution will run for

    Returns:
        (pd.DataFrame): dataframe object of the baseline results.

    Raises:
        None.
    """
    # define & initialize output
    baseline_evaluation_data = []

    # run through the number of trials
    for each_episode in range(num_episodes):
        # creates a fresh environment each run => this is a must to get
        # consistent data
        env = env_instance()

        # reset first before starting
        observations, info = env.reset()

        # define container flags to break from the while loop
        is_terminated = False
        is_truncated = False

        # hold the current trial data and then log it for each time it
        # completes
        current_episode_data = []

        # track against the duration value => how long do you want to run this.
        step_counter = 0

        # calculate the total points from all trial runs
        total_reward_points = 0

        # while these flags are not true and the counter is less than the
        # max_step do ->
        while (not is_terminated and not is_truncated) and (step_counter <
                                                            SIMULATED_DURATION_STEPS):
            # action passed assumes there is not RL influence because the step()
            # calculates the final_value (hybrid)
            action_without_rl_influence = np.array([0.0])

            # create a tuple for the output of the step() => this outputs a
            # tuple once the action is passed & executed
            observations, reward, is_terminated, is_truncated, info = env.step(
                action_without_rl_influence)

            # update trackers
            total_reward_points += reward
            step_counter += 1

            # logging
            step_log = info.copy()
            step_log['reward'] = reward
            step_log['episode'] = each_episode
            step_log['step'] = step_counter
            step_log['time'] = step_counter * DELTA_T_IN_SEC

            current_episode_data.append(step_log)

        # once the while loop breaks, add all the trials into the main variable
        # and return the data list
        # extend: add one list to another
        baseline_evaluation_data.extend(current_episode_data)
        env.close()

    return pd.DataFrame(baseline_evaluation_data)
