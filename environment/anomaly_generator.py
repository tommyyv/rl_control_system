# user-defined modules
from config import (
    ANOMALY_MAGNITUDE_MIN,
    ANOMALY_MAGNITUDE_MAX)


class AnomalyGenerator():
    """AnomalyGenerator Class object.

    Attributes:
        start_min (float): minimum step counter to trigger the anomaly effect
        start_max (float): maximum step counter to cap off the anomaly effect
        duration_min (float): minimum duration for the effect to last
        duration_max (float): maximum duration for the effect to last
        magnitude_min (float): minimum strength of the effect
        magnitude_max (float): maximum strength of the effect

    Methods:
        get_anomaly_effect (tuple[float, bool]): returns an anomaly effect used
            in the training process
        reset: resets the anomaly generator
    """

    def __init__(self,
                 start_min: float,
                 start_max: float,
                 duration_min: float,
                 duration_max: float,
                 magnitude_min: float,
                 magnitude_max: float):
        self.start_min = start_min
        self.start_max = start_max
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.magnitude_min = magnitude_min
        self.magnitude_max = magnitude_max
        self.anomaly_details = None

    def get_anomaly_effect(self, current_step: int) -> tuple[float, bool]:
        """Generates the anomaly effects used during the model training phase.

        Args:
            current_step (int): integer step counter to track when the anomaly
                takes effect.

        Returns:
            (tuple[float, bool]): the generated anomaly value and triggers state
                for the is_anomaly_active variable.

        Raises:
            None.
        """
        if self.anomaly_details and current_step >= self.anomaly_details['start_step'] and current_step < self.anomaly_details['start_step'] + self.anomaly_details['duration']:
            return self.anomaly_details['magnitude'], True
        return 0.0, False

    def reset(self, np_random: float):
        """Resets the AnomalyGenerator class instance.

        Args:
            np_random (float): generates a random number using NumPy Generator
            Class object.

        Returns:
            None.

        Raises:
            None.
        """
        magnitude = np_random.uniform(
            ANOMALY_MAGNITUDE_MIN, ANOMALY_MAGNITUDE_MAX)
        start_step = np_random.integers(self.start_min, self.start_max)
        duration = np_random.integers(
            self.duration_min, self.duration_max)

        if np_random.random() < 0.5:  # rand() between 0.0 and 1.0
            # multiple the magnitude by -1 if the random value is less than 0.5
            magnitude *= -1

        self.anomaly_details = {
            'start_step': start_step,
            'magnitude': magnitude,
            'duration': duration
        }
