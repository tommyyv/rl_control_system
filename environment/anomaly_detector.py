# python libraries and modules
import collections
import numpy as np


class AnomalyDetector():
    """AnomalyDetector Class object.

    Attributes:
        anomaly_detection_threshold (float):
        window_size (int):

    Methods:
        detect_anomaly (bool): checks if an anomaly has been detected
        reset: resets the class instance
    """

    def __init__(self,
                 anomaly_detection_threshold: float,
                 window_size: int):
        self.is_anomaly_active = False
        self.anomaly_detection_threshold = anomaly_detection_threshold
        self.window_size = window_size
        self.recent_errors = collections.deque(maxlen=window_size)

    def detect_anomaly(self, current_error: float) -> bool:
        """Detects if an anomaly is present.

        Args:
            current_error (float): calculated error value

        Returns:
            (bool): checks if an anomaly is present

        Raises:
            None.
        """
        current_error = abs(current_error)
        self.recent_errors.append(current_error)

        if len(self.recent_errors) < self.window_size:
            self.is_anomaly_active = False
            return self.is_anomaly_active

        average_window_error = np.mean(self.recent_errors)

        if (current_error > self.anomaly_detection_threshold) and (average_window_error > (self.anomaly_detection_threshold *
                                                                                           0.75)):
            self.is_anomaly_active = True
        else:
            self.is_anomaly_active = False

        return self.is_anomaly_active

    def reset(self):
        """Resets the AnomalyDetector instance.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.recent_errors.clear()
        self.is_anomaly_active = False
