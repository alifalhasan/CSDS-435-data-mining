import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Evaluator:
    def __init__(self, n_folds=10, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

        self.scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }

    def cross_validate(self, model, X, y):
        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scores = cross_validate(
            model.model,
            X_scaled,
            y,
            cv=cv,
            scoring=self.scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        return {
            "accuracy": np.mean(scores["test_accuracy"]),
            "precision": np.mean(scores["test_precision"]),
            "recall": np.mean(scores["test_recall"]),
            "f1_score": np.mean(scores["test_f1"]),
        }