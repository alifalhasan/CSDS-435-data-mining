from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=0.001,  # L2 regularization
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=200,
            random_state=random_state,
            early_stopping=True,  # Overfitting atkanor jonno
            n_iter_no_change=10,  # Already doing good enough
            validation_fraction=0.1,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
