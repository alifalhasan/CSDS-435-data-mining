from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


class DecisionTree:
    def __init__(self, max_depth=3, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            criterion="gini",
            splitter="best",
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
