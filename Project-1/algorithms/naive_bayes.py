from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
