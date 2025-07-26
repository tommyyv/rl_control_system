import numpy as np

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from data_processing.create_dataframe import create_dataframe
from data_processing.dataset_loader import DatasetLoader

from config import (
    MLP_HIDDEN_LAYER_SIZES,
    MLP_ACTIVATION,
    MLP_ALPHA,
    MLP_LEARNING_RATE_INIT,
    MLP_SOLVER,
    MLP_N_ITER,
    DATASET_PATH
)


def optimize_mlp_hyperparameters():
    """
    Performs RandomizedSearchCV to find optimal hyperparameters for the SystemIdentifier model.
    Prints the best parameters and performance metrics.

    Args:
        data_filepath (str): The path to the raw dataset CSV file.
    """
    print("--- Starting SystemIdentifier Hyperparameter Tuning (RandomizedSearchCV) ---")

    dataset_loader = DatasetLoader(DATASET_PATH)
    dataset = dataset_loader.load_data()

    # 1. Load and Clean Data
    dataframe = create_dataframe(dataset)
    if dataframe.empty:
        print("Tuning aborted: DataFrame is empty after cleaning.")
        return

    # 2. Prepare Data for SystemIdentifier Training
    dataframe_target_col = 'Return Air Temperature'
    dataframe_data_cols_list = ['Supply Air Temperature',
                                'Return Air Temperature',
                                'Outdoor Air Temperature']

    df_processed = dataframe.copy()
    next_target_col_name = f'next_{dataframe_target_col}'
    df_processed[next_target_col_name] = df_processed[dataframe_target_col].shift(
        -1)
    df_processed.dropna(inplace=True)

    x_values = df_processed[dataframe_data_cols_list].values
    y_values = df_processed[next_target_col_name].values

    # 3. Split data into training and testing sets for SystemIdentifier validation
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Define the estimator and parameter distributions for RandomizedSearchCV
    estimator = None
    param_distributions = {}

    best_estimator = None
    best_params = {}
    best_score = -np.inf

    estimator = MLPRegressor(
        max_iter=2000, random_state=42)  # Base MLPRegressor

    param_distributions = {
        'hidden_layer_sizes': MLP_HIDDEN_LAYER_SIZES,
        'activation': MLP_ACTIVATION,
        'solver': MLP_SOLVER,
        'alpha': MLP_ALPHA,
        'learning_rate_init': MLP_LEARNING_RATE_INIT
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=MLP_N_ITER,  # Number of random combinations to try
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',  # Optimize for lower MSE
        n_jobs=-1,  # Use all available CPU cores
        verbose=2,  # Verbosity level
        random_state=42  # For reproducibility of search
    )
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)

    # 5. Perform the Hyperparameter Search
    random_search.fit(x_train, y_train)

    # 6. Get the Best Estimator and Parameters
    best_estimator = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # 7. Print Tuning Results
    print("\n--- SystemIdentifier Tuning Results ---")
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation score (negative MSE): {best_score:.4f}")

    # Evaluate the Best Model on the held-out test set
    test_predictions = best_estimator.predict(x_test)
    test_mse = mean_squared_error(y_test, test_predictions)

    # 8. Output Recommended Parameters for config.py (MANUAL UPDATE REQUIRED)
    print("\n--- RECOMMENDED CONFIG.PY UPDATES (Copy these values) ---")
    for param, value in best_params.items():
        if isinstance(value, tuple):  # For hidden_layer_sizes
            print(f"MLP_{param.upper()}_OPTIMAL = {value}")
        else:
            # Use !r for correct string representation
            print(f"MLP_{param.upper()}_OPTIMAL = {value!r}")
    print("-----------------------------------------------------")


optimize_mlp_hyperparameters()
