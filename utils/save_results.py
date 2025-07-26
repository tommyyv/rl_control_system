# python libraries and modules
import os
import datetime


def save_results(results_dir: str) -> str:
    """
    Saves and formats the results directory name.

    Args:
        results_dir (str): results directory filepath.

    Returns:
        (str): formatted directory
            Example: YYYYMMDD_THH:MM_results

    Raises:
        None.
    """
    date_format = datetime.datetime.now().strftime("%Y%m%d_T%H:%M")
    final_path = os.path.join(results_dir, date_format) + '_results'
    os.makedirs(final_path, exist_ok=True)

    return final_path
