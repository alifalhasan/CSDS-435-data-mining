import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_and_preprocess(filename, data_dir="data"):
    try:
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy="mean")

        data = pd.read_csv(f"{data_dir}/{filename}", sep="\t", header=None)
        gene_ids = data.iloc[:, 0].values
        true_labels = data.iloc[:, 1].values
        features = data.iloc[:, 2:].values

        print(f"Loaded {filename}:")
        print(f"  - Samples: {len(gene_ids)}")
        print(f"  - Features: {features.shape[1]}")
        print(f"  - True clusters: {len(np.unique(true_labels[true_labels != -1]))}")
        print(f"  - Outliers: {np.sum(true_labels == -1)}")

        # Remove constant columns
        non_constant_cols = ~(
            np.all(features == features[0, :], axis=0) | np.all(features == 0, axis=0)
        )
        features = features[:, non_constant_cols]

        # Handle missing values and standardize
        features = imputer.fit_transform(features)
        features_scaled = scaler.fit_transform(features)

        return {
            "gene_ids": gene_ids,
            "true_labels": true_labels,
            "features": features_scaled,
            "dataset_name": filename.replace(".txt", ""),
        }

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
