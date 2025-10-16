from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVM:
    def __init__(self, C=1.0, kernel="rbf", random_state=42):
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma="scale",
            random_state=random_state,
            probability=False,
            cache_size=512,
            max_iter=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
