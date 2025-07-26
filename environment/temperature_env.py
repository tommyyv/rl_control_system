#python libraries & modules
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# user-defined modules
from controllers.pid_controller import PIDController
from environment.anomaly_detector import AnomalyDetector
from environment.anomaly_generator import AnomalyGenerator
from config import (
    DELTA_T_IN_SEC,
    PID_KP,
    PID_KI,
    PID_KD,
    TEMP_SETPOINT,
    SETPOINT_MIN_FALLBACK,
    SETPOINT_MAX_FALLBACK,
    OUTDOOR_AIR_TEMP_MIN_FALLBACK,
    OUTDOOR_AIR_TEMP_MAX_FALLBACK,
    OUTDOOR_AIR_TEMP_MIN_WALK_CLAMP,
    OUTDOOR_AIR_TEMP_MAX_WALK_CLAMP,
    INITIAL_TEMP_DEVIATION_RANGE,
    SUPPLY_AIR_TEMP_MIN_LIMIT,
    SUPPLY_AIR_TEMP_MAX_LIMIT,
    CRITICAL_TEMP_LOW,
    CRITICAL_TEMP_HIGH,
    SIMULATED_DURATION_STEPS,
    REWARD_CRITICAL_VIOLATION_PENALTY,
    REWARD_CONTROL_EFFORT_PENALTY,
    REWARD_ERROR_PENALTY,
    REWARD_ANOMALY_CORRECTED_REWARD,
    REWARD_TIMEOUT_PENALTY,
    RECOVERY_TIME_LIMIT,
    NOISE_MAGNITUDE,
    ANOMALY_MAGNITUDE_MIN,
    ANOMALY_MAGNITUDE_MAX,
    ANOMALY_DURATION_MIN,
    ANOMALY_DURATION_MAX,
    ANOMALY_START_MIN,
    ANOMALY_START_MAX,
    ANOMALY_DETECTION_THRESHOLD,
    ANOMALY_DETECTION_WINDOW
)


class TemperatureControlEnv(gym.Env):
    '''Temperature Control Environment Class object.

    Environment used to train model.

    Attributes:
        system_model ()
        system_data_cols ():
        outdoor_air_temp_source ():
        setpoint_source ():
        current_supply_air_temp ():

    Methods:
        step (tuple[np.ndarray, float, bool, bool, dict]):
        _get_info (dict):
        _get_obs (np.ndarray):
        reset: 
    '''

    def __init__(self,
                 system_model=None,
                 setpoint_source=None,
                 outdoor_air_temp_source=None,
                 supply_air_temp_source=None):
        super().__init__()
        self.system_model = system_model

        self.system_data_cols = system_model.data_cols

        self.outdoor_air_temp_source = outdoor_air_temp_source
        self.setpoint_source = setpoint_source
        self.setpoint_dynamic = (self.setpoint_source is not None and
                                 len(self.setpoint_source) > 0)
        self.current_supply_air_temp = supply_air_temp_source

        # -- observation (input) space: what the RL agent receives from the env --
        # observation at the lowest bound:
        # return_air_temp_error, return_air_temp_delta_error, SAT,
        # time_since_anomaly, flag, outdoor_air_temp, setpoint
        obs_low = np.array([-30.0,
                            -30.0,
                           SUPPLY_AIR_TEMP_MIN_LIMIT,
                           0.0,
                           0.0,
                           OUTDOOR_AIR_TEMP_MIN_FALLBACK,
                           SETPOINT_MIN_FALLBACK],
                           dtype=np.float32)

        # observation at the highest bound
        obs_high = np.array([30.0,
                             30.0,
                             SUPPLY_AIR_TEMP_MAX_LIMIT,
                             SIMULATED_DURATION_STEPS,
                             1.0,
                             OUTDOOR_AIR_TEMP_MAX_FALLBACK,
                             SETPOINT_MAX_FALLBACK],
                            dtype=np.float32)

        # shape means how many distinctive features
        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            dtype=np.float32)

        # -- action (output) space: what the RL agent sends to the env --
        # bound checks the adjustment factor: -1.0 to 1.0
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(1,),
                                       dtype=np.float32)

        # -- local state variables --
        self.current_return_air_temp = TEMP_SETPOINT
        self.time_step_counter = 0
        self.previous_error = 0.0
        self.previous_sat_control_signal = 0.0
        self.current_outdoor_temp = 0.0
        self.current_setpoint = TEMP_SETPOINT
        self.reward = 0.0

        self.outdoor_air_temp_idx = 0
        self.setpoint_idx = 0

        # -- helper modules object instances --
        self.pid_controller = PIDController(PID_KP,
                                            PID_KI,
                                            PID_KD,
                                            SUPPLY_AIR_TEMP_MIN_LIMIT,
                                            SUPPLY_AIR_TEMP_MAX_LIMIT)

        self.anomaly_generator = AnomalyGenerator(ANOMALY_START_MIN,
                                                  ANOMALY_START_MAX,
                                                  ANOMALY_DURATION_MIN,
                                                  ANOMALY_DURATION_MAX,
                                                  ANOMALY_MAGNITUDE_MIN,
                                                  ANOMALY_MAGNITUDE_MAX)

        self.anomaly_detector = AnomalyDetector(
            ANOMALY_DETECTION_THRESHOLD, ANOMALY_DETECTION_WINDOW)

        # -- tracking anomaly for current episode --
        self.current_anomaly_magnitude = 1.0
        self.anomaly_start_time = -1
        self.time_since_anomaly_start = 0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool,
                                                dict]:
        """Main driver of the Temperature Environment; receives an action to
        respond to.

        Executes an episode run using the previous observation.

        Args:
            action (np.ndarray): the input observation

        Returns:
            (tuple[np.ndarray, float, bool, bool, dict]):

        Raises:
            None.
        """
        # observation element in the tuple
        rl_adjustment_factor = action[0]

        # define & initialize variables
        # -- update setpoint: source or fallback value --
        if self.setpoint_dynamic:
            self.setpoint_idx = (self.setpoint_idx +
                                 1) % len(self.setpoint_source)
            self.current_setpoint = self.setpoint_source[self.setpoint_idx]
        else:
            self.current_setpoint = self.np_random.uniform(
                SETPOINT_MIN_FALLBACK, SETPOINT_MAX_FALLBACK)

        # -- update current return temp --
        self.current_return_air_temp = self.current_setpoint + \
            self.np_random.uniform(-INITIAL_TEMP_DEVIATION_RANGE,
                                   INITIAL_TEMP_DEVIATION_RANGE)

        # -- update outdoor temperature --
        if self.outdoor_air_temp_source is not None and len(self.outdoor_air_temp_source) > 0:
            # If a source is provided, advance the index and get the next value from the source
            self.outdoor_air_temp_idx = (
                self.outdoor_air_temp_idx + 1) % len(self.outdoor_air_temp_source)
            self.current_outdoor_temp = self.outdoor_air_temp_source[self.outdoor_air_temp_idx]
        else:
            # steadily randomly walk the outdoor temp
            self.current_outdoor_temp += self.np_random.normal(0,
                                                               NOISE_MAGNITUDE)
            self.current_outdoor_temp = np.clip(self.current_outdoor_temp,
                                                OUTDOOR_AIR_TEMP_MIN_WALK_CLAMP,
                                                OUTDOOR_AIR_TEMP_MAX_WALK_CLAMP)

        # -- disturbances --
        disturbance_input = self.np_random.normal(0, NOISE_MAGNITUDE)

        anomaly_magnitude, is_anomaly_detected = self.anomaly_generator.get_anomaly_effect(
            self.time_step_counter)

        disturbance_input += anomaly_magnitude

        if is_anomaly_detected:
            # if start_time is not enabled then use the time_step_counter
            if self.anomaly_start_time == -1:
                self.anomaly_start_time = self.time_step_counter
            # otherwise find how long it's been using the current step_counter -
            # the start time set
            self.time_since_anomaly_start = self.time_step_counter - self.anomaly_start_time
        # otherwise, disabled both
        else:
            self.anomaly_start_time = -1
            self.time_since_anomaly_start = 0

        # init error calculation
        current_error = self.current_setpoint - self.current_return_air_temp

        # u = control input, y = control output measurement
        u_pid = self.pid_controller.calculate_control_signal(
            self.current_setpoint,
            self.current_return_air_temp,
            DELTA_T_IN_SEC)

        self.anomaly_detector.detect_anomaly(abs(current_error))

        # u_rl_adjustment is the action output from the rl ppo algorithm
        # the ppo algo should receive inputs (observations & rewards) to
        # generate an action
        u_rl_correction = 0.0

        # anomaly exists then setup the correct values
        if self.anomaly_detector.is_anomaly_active:
            # if there is an anomaly then set SAT_magnitude (deviation) to
            max_saturation_correction_magnitude = (SUPPLY_AIR_TEMP_MAX_LIMIT -
                                                   SUPPLY_AIR_TEMP_MIN_LIMIT) / 2

            # rl_adjustment_factor is action[0], which should be the observation
            u_rl_correction = rl_adjustment_factor * max_saturation_correction_magnitude

        u_final_supply_air_temp = u_pid + u_rl_correction

        u_final_supply_air_temp = np.clip(u_final_supply_air_temp,
                                          SUPPLY_AIR_TEMP_MIN_LIMIT,
                                          SUPPLY_AIR_TEMP_MAX_LIMIT)

        u_previous_sat_control_signal = u_final_supply_air_temp

        # -- update system dynamics --
        # this updates the system identifier columns and add the value to the
        # each respective column
        data_cols_for_system_model = []
        for each_col_name in self.system_data_cols:
            if each_col_name == 'Return Air Temperature':
                data_cols_for_system_model.append(self.current_return_air_temp)
            elif each_col_name == 'Supply Air Temperature':
                data_cols_for_system_model.append(u_final_supply_air_temp)
            elif each_col_name == 'Outdoor Air Temperature':
                data_cols_for_system_model.append(self.current_outdoor_temp)

        predicted_next_state = self.system_model.predict_next_state(
            data_cols_for_system_model)

        self.current_return_air_temp = predicted_next_state + disturbance_input

        # -- update time & error --
        self.time_step_counter += 1
        self.previous_error = self.current_setpoint - self.current_return_air_temp

        # -- reward calculation --
        reward = 0.0

        reward += REWARD_ERROR_PENALTY * abs(current_error)
        reward += REWARD_CONTROL_EFFORT_PENALTY * abs(u_final_supply_air_temp)

        terminated = False

        if not (CRITICAL_TEMP_LOW <= self.current_return_air_temp <=
                CRITICAL_TEMP_HIGH):
            reward += REWARD_CRITICAL_VIOLATION_PENALTY
            terminated = True

        if self.anomaly_detector.is_anomaly_active and is_anomaly_detected:
            if (abs(current_error) < ANOMALY_DETECTION_THRESHOLD) / 2:
                reward += REWARD_ANOMALY_CORRECTED_REWARD
            elif self.time_since_anomaly_start > RECOVERY_TIME_LIMIT and (abs(current_error) < ANOMALY_DETECTION_THRESHOLD):
                reward += REWARD_TIMEOUT_PENALTY

        self.reward = reward

        # -- terminate & truncation conditions --
        # this is a failure state
        terminated = not (CRITICAL_TEMP_LOW <=
                          self.current_return_air_temp <= CRITICAL_TEMP_HIGH)

        # truncated = run out of time or reach max steps
        truncated = self.time_step_counter >= SIMULATED_DURATION_STEPS

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Gets the observation of the environment.

        Args:
            None.

        Returns:
            (np.ndarray): Returns an array of the current observation.

        Raises:
            None.
        """
        return_air_temp_error = self.current_return_air_temp - self.current_setpoint
        return_air_temp_delta_error = (return_air_temp_error - self.previous_error) / \
            DELTA_T_IN_SEC if self.time_step_counter > 0 else 0.0
        is_anomaly_detected = float(self.anomaly_detector.is_anomaly_active)

        return np.array([
            self.previous_sat_control_signal,
            self.current_return_air_temp,
            self.current_outdoor_temp,
            self.current_setpoint,
            return_air_temp_error,
            return_air_temp_delta_error,
            is_anomaly_detected,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        """Gets the information object of the environment.

        Args:
            None.

        Returns:
            (dict): information object returned used for observations

        Raises:
            None.
        """
        return {
            'current_supply_air_temp': self.current_supply_air_temp,
            'current_return_air_temp': self.current_return_air_temp,
            'current_outdoor_temp': self.current_outdoor_temp,
            'current_setpoint': self.current_setpoint,
            'previous_sat_control_signal': self.previous_sat_control_signal,
            'is_anomaly_detected': self.anomaly_detector.is_anomaly_active,
            'return_air_temp_error': self.current_return_air_temp - self.current_setpoint,
        }

    def reset(self,
              seed: int = None,
              options: dict = None) -> tuple[np.ndarray, dict]:
        """Resets the TemperatureControlEnv Class object.

        Args:

        Returns:
            None.

        Raises:
            None.
        """
        super().reset(seed=seed)

        # -- reset state variables -- => take whatever is in __init__ and zero
        self.previous_sat_control_signal = TEMP_SETPOINT
        self.time_step_counter = 0
        self.reward = 0.0

        # -- reset modules --
        self.pid_controller.reset()
        self.anomaly_generator.reset(self.np_random)
        self.anomaly_detector.reset()

        # -- reset anomaly tracker --
        self.current_anomaly_magnitude = 0.0
        self.anomaly_start_time = -1
        self.time_since_anomaly_start = 0

        # -- reset setpoint: source or fallback value --
        if self.setpoint_dynamic:
            self.setpoint_idx = self.np_random.integers(0,
                                                        len(self.setpoint_source))
            self.current_setpoint = self.setpoint_source[self.setpoint_idx]
        else:
            self.current_setpoint = self.np_random.uniform(
                SETPOINT_MIN_FALLBACK, SETPOINT_MAX_FALLBACK)

        # -- reset current return temp --
        self.current_return_air_temp = self.current_setpoint + \
            self.np_random.uniform(-INITIAL_TEMP_DEVIATION_RANGE,
                                   INITIAL_TEMP_DEVIATION_RANGE)

        # -- reset outdoor temperature --
        if self.outdoor_air_temp_source is not None and len(self.outdoor_air_temp_source) > 0:
            self.outdoor_air_temp_idx = self.np_random.integers(0,
                                                                len(self.outdoor_air_temp_source))
            self.current_outdoor_temp = self.outdoor_air_temp_source[self.outdoor_air_temp_idx]
        else:
            # steadily randomly walk the outdoor temp
            self.current_outdoor_temp = self.np_random.uniform(
                OUTDOOR_AIR_TEMP_MIN_FALLBACK, OUTDOOR_AIR_TEMP_MAX_FALLBACK)

        self.prev_error = self.current_setpoint - self.current_return_air_temp

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        pass
