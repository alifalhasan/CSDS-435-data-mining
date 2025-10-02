from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNN:
    def __init__(self, n_neighbors=7):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm="auto", weights="uniform", n_jobs=-1
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
