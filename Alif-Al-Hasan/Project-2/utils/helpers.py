import os
import random
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# Handles the result folder
def create_results_folder(results_dir):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "pca"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tsne"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "numerical"), exist_ok=True)

    return results_dir


# Optimal k for k-means
def determine_optimal_k(features, max_k=15):
    if len(features) < 3:
        return 2

    max_k = min(max_k, len(features) - 1)
    k_range = range(2, max_k)

    best_silhouette = -1
    best_k = 2

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(features, labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_k = k
        except:
            continue

    return best_k


# Optimal epsilon for dbscan
def determine_dbscan_eps(features, k=5):
    if len(features) <= k:
        return 0.5

    try:
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(features)
        distances, _ = neighbors_fit.kneighbors(features)
        distances = np.sort(distances[:, k - 1], axis=0)

        if len(distances) > 10:
            eps = np.percentile(distances, 90)
        else:
            eps = 0.5

        return max(0.3, min(eps, 2.0))
    except:
        return 0.5
