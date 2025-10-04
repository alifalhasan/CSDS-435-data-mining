import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(data_path):
    data = pd.read_csv(data_path, header=None, sep=r"\s+", engine="python")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    return X.values, y.values


class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=[256, 128, 64, 32], dropout_rate=0.3):
        super(AdvancedNeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_layers):
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.extend([nn.Linear(prev_size, 1)])

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x).squeeze()


class PyTorchNeuralNetwork:
    def __init__(
        self,
        hidden_layers=[256, 128, 64, 32],
        learning_rate=0.001,
        epochs=200,
        batch_size=32,
        dropout_rate=0.3,
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_model(self, input_size):
        self.model = AdvancedNeuralNetwork(
            input_size=input_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        if self.model is None:
            self.reset_model(X.shape[1])

        # Ensure batch size is at least 2
        effective_batch_size = max(2, min(self.batch_size, len(X)))

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        criterion = nn.BCEWithLogitsLoss()

        self.model.train()

        for epoch in range(self.epochs):
            epoch_losses = []

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Ensure proper shapes
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_y.dim() == 0:
                    batch_y = batch_y.unsqueeze(0)

                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_losses.append(loss.item())

            avg_epoch_loss = np.mean(epoch_losses)
            scheduler.step(avg_epoch_loss)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}")

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs)
            return (probabilities.cpu().numpy() > 0.5).astype(int)


def evaluate_model(model, X, y, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_folds}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.reset_model(X.shape[1])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
    }


def main():
    datasets = {
        "Dataset1": "data/project1_dataset1.txt",
        "Dataset2": "data/project1_dataset2.txt",
    }

    model = PyTorchNeuralNetwork(
        hidden_layers=[256, 128, 64, 32],
        learning_rate=0.001,
        epochs=200,
        batch_size=32,
        dropout_rate=0.3,
    )

    for name, path in datasets.items():
        X, y = load_data(path)
        print(f"\n--- Processing {name} ---")

        # Evaluate model
        scores = evaluate_model(model, X, y, n_folds=10)

        # Print results
        print(f"  Accuracy:  {scores['accuracy']:.4f}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1-Score:  {scores['f1_score']:.4f}")


if __name__ == "__main__":
    main()
