import numpy as np
from sklearn.neural_network import MLPRegressor

from config import (
    MLP_HIDDEN_LAYER_SIZES_OPTIMAL,
    MLP_ACTIVATION_OPTIMAL,
    MLP_SOLVER_OPTIMAL,
    MLP_ALPHA_OPTIMAL,
    MLP_LEARNING_RATE_INIT_OPTIMAL,
)


class SystemIdentifier():
    """System identifier Class object.

    Learns, understands, and trains the data model to predict the next set of
    values before a reinforcement learning algorithm is applied.

    Attributes:
        model (sklearn.neural_network.MLPRegressor): MLPRegressor Class object using optimized hyperparameters
        data_cols (list[str]): input columns
        target_col (str): output column

    Methods:
        train_model: returns a trained model using a multi-layer regression
        algorithm
        predict_next_value: predicts the next state used by the model within the
        training environment
    """

    def __init__(self):
        self.model = MLPRegressor(random_state=42,
                                  max_iter=2000,
                                  hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES_OPTIMAL,
                                  activation=MLP_ACTIVATION_OPTIMAL,
                                  solver=MLP_SOLVER_OPTIMAL,
                                  alpha=MLP_ALPHA_OPTIMAL,
                                  learning_rate_init=MLP_LEARNING_RATE_INIT_OPTIMAL)
        self.data_cols = None
        self.target_col = None

    def train_model(self, dataframe, target_col, data_cols) -> 'SystemIdentifier':
        """Trains the data model using MLP Regressor.

        Args:
            dataframe (pd.DataFrame): data
            target_col (str): what columns to look at => output col
            data_cols (list[str]): input columns

        Returns:
            (SystemIdentifier): The best fit line.

        Raises:
            None.
        """
        self.data_cols = data_cols
        self.target_col = target_col

        dataframe_copy = dataframe.copy()

        next_target_col = f'next_{target_col}'
        # shift(-1) = gives next value
        dataframe_copy[next_target_col] = dataframe_copy[target_col].shift(-1)
        dataframe_copy.dropna(inplace=True)

        x_axis = dataframe_copy[data_cols].values  # data_col
        y_axis = dataframe_copy[next_target_col].values  # target_col

        self.model.fit(x_axis, y_axis)

        # predictions = self.model.predict(x_axis)
        # mse = mean_squared_error(y_axis, predictions)
        # print(f"SystemIdentifier training MSE on in-sample data: {mse:.4f}")

        return self

    def predict_next_state(self, data_col_arr: np.ndarray) -> float:
        """Predicts the next value of the dataset.

        Args:
            data_col_arr (np.ndarray): 1-D array of the input feature column.

        Returns:
            (float): The next predicted value, given an reshaped array.

        Raises:
            None.

        """
        return self.model.predict(np.array(data_col_arr).reshape(1, -1))[0]
