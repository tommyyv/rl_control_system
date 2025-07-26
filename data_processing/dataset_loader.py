import pandas as pd


class DatasetLoader():
    """Dataset Loader Class object.

    Loads dataset, given a filepath.

    Attributes:

    Methods:

    """
    filepath: str
    data: dict

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load_data(self) -> dict:
        """Loads data from a CSV file.

        Args:
            None.

        Returns:
            (pd.DataFrame): dataset from CSV file.

        Raises:
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Dataset loaded from: {self.filepath}")
            print(f"Dataset shape: {self.data.shape}")
            print(f"Dataset columns: {self.data.columns.tolist()}")
            # Basic preprocessing: ensure timestamp is datetime, sort by time
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data = self.data.sort_values(
                    'timestamp').set_index('timestamp')

            return self.data
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.filepath}")
            return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
