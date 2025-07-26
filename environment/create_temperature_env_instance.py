# python libraries & modules
from typing import Type
import numpy as np

# user-defined modules
from environment.temperature_env import TemperatureControlEnv
from data_processing.system_identifier import SystemIdentifier


def create_temperature_env_instance(system_model: Type[SystemIdentifier],
                                    outdoor_air_temp_source: np.ndarray,
                                    setpoint_source: np.ndarray,
                                    supply_air_temp_source: np.ndarray) -> 'TemperatureControlEnv':
    """Builder pattern for creating a TemperatureControlEnv Class object.

        Args:
            system_model (SystemIdentifier):
            outdoor_air_temp_source ()
            setpoint_source ():
            supply_air_temp_source ():

        Returns:
            generated TemperatureControlEnv class instance
        Raises:
            None.
    """
    return TemperatureControlEnv(
        system_model=system_model,
        outdoor_air_temp_source=outdoor_air_temp_source,
        setpoint_source=setpoint_source,
        supply_air_temp_source=supply_air_temp_source
    )
