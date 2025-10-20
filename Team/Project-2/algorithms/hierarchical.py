import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils.helpers import determine_optimal_k


def hierarchical_clustering(features, config, linkage="ward", true_labels=None):
    if true_labels is not None and len(np.unique(true_labels[true_labels != -1])) > 0:
        n_clusters = len(np.unique(true_labels[true_labels != -1]))
    else:
        n_clusters = determine_optimal_k(features, config["max_k_search"])

    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, metric="euclidean"
    )
    labels = hierarchical.fit_predict(features)

    details = {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "algorithm": "Hierarchical",
    }

    return labels, details
