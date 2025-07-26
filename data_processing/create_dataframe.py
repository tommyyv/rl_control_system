import pandas as pd


def create_dataframe(dataset: str) -> pd.DataFrame:
    """Creates a DataFrame Class object.

    Args:
        dataset (str): loaded CSV dataset.

    Returns:
        (pd.DataFrame): processed DataFrame object with targeted columns.

    Raises:
        None.
    """
    dataframe_columns = [
        'Supply Air Temperature',
        'Return Air Temperature',
        'Supply Air Temperature Setpoint',
        'Outdoor Air Temperature'
    ]

    dataframe = pd.DataFrame(data=dataset, columns=dataframe_columns)

    return dataframe
