# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
import numpy as np
from dir_config import DirConfig
import pandas as pd
import numpy as np

## ============================================================
from main_scripts.dir_config import DirConfig

class TrainTestSplit:
    """A class to handle train-test splitting of datasets."""
    def __init__(self, test_size=0.2, seed=42, pipe_path=DirConfig().load_meteo_data(), pm25_path="pm25"):
        self.test_size = test_size
        self.seed = seed
        np.random.seed(seed)
        self.pm25_path = pm25_path
        self.pipe_path = pipe_path

    def time_split(self, df=None, target=None, test_size=None):
        """Splits the DataFrame into training and testing sets based on time."""
        if test_size is None:
            test_size = self.test_size
        if target is None:
            target = self.pm25_path
        if df is None:
            df = pd.read_csv(self.pipe_path / self.pm25_path)
            print("train-test split, df shape:", df.shape)

        n = len(df)        
        n_test = int(np.ceil(n * self.test_size))
        
        train = df.iloc[:-n_test]      
        test = df.iloc[-n_test:]

        X_train = train.drop(columns=[target])
        y_train = train[target]
        X_test = test.drop(columns=[target])
        y_test = test[target]

        print(f"Train rows: {len(train)}, Test rows: {len(test)}")
        return X_train, X_test, y_train, y_test, train, test
    