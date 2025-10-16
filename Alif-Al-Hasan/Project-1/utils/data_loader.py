import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def load_data(self, data_path):
        data = pd.read_csv(data_path, header=None, sep=r"\s+", engine="python")

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_processed = self._preprocess_features(X)

        return X_processed.values, y.values

    def _preprocess_features(self, X):
        X_processed = X.copy()

        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X[col])

        return X_processed
