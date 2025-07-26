class PIDController():
    """PID controller Class object.

    Attributes:
        kp (float): determines how aggressive the controller responds based on error
        ki (float): determines how much correction is needed over time
        kd (float): determines how much error has changed over time
        integral (float):
        prev_err (float):

    Methods:
        calculate_control_signal (float): returns a control signal used in the
        baseline and hybrid evaluations.
        reset: resets and zeros out the PID controller variables.
    """
    kp: float
    ki: float
    kd: float
    integral: float
    prev_err: float

    def __init__(self,
                 kp: float,
                 ki: float,
                 kd: float,
                 sat_min_limit: float,
                 sat_max_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_err = 0.0
        self.sat_min_limit = sat_min_limit
        self.sat_max_limit = sat_max_limit

    def calculate_control_signal(self, setpoint: float,
                                 curr_temp_reading: float,
                                 delta_t: float) -> float:
        """Calculates a control signal based on current error, integral, and
        derivative.

        Args:
            setpoint (float): supply air temperature setpoint
            curr_temp_reading (float): current return air temperature reading
            delta_t (float): time measurement in seconds

        Returns:
            (float): calculated control signal

        Raises:
            None.

        """
        err = setpoint - curr_temp_reading

        derivative = (err - self.prev_err) / delta_t if delta_t > 0 else 0.0

        unclipped_output = (self.kp * err) + self.ki * (self.integral
                                                                  + err * delta_t) + (self.kd * derivative)

        if (unclipped_output > self.sat_min_limit and unclipped_output < self.sat_max_limit) or \
           (unclipped_output >= self.sat_max_limit and err < 0) or \
           (unclipped_output <= self.sat_min_limit and err > 0):
            self.integral += err * delta_t

        self.prev_err = err

        return (self.kp * err) + (self.ki * self.integral) + (self.kd *
                                                              derivative)

    def reset(self):
        """Resets the PID controller.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.integral = 0.0
        self.prev_err = 0.0
