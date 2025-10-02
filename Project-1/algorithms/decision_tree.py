from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


class DecisionTree:
    def __init__(self, max_depth=7, random_state=42):
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

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        return self.model.feature_importances_  # Retrieve important features
