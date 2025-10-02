import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def load_data(self, data_path):
        data = pd.read_csv(data_path, header=None, sep=r'\s+', engine='python')

        # Convert to numpy array
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        X = self._fast_preprocess(X)

        return X, y
    
    def _fast_preprocess(self, X):
        if X.dtype == object:
            # Use LabelEncoder for categorical data
            X_encoded =  np.zeros_like(X, dtype=float)
            for col in range(X.shape[1]):
                if X[:, col].dtype == object:
                    le = LabelEncoder()
                    X_encoded[:, col] = le.fit_transform(X[:, col])
                else:
                    X_encoded[:, col] = X[:, col]
            return X_encoded
        return X