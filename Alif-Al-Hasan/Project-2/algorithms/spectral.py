from sklearn.cluster import SpectralClustering
from utils.helpers import determine_optimal_k


def spectral_clustering(features, config, true_labels=None):
    optimal_k = determine_optimal_k(features, config["max_k_search"])

    spectral = SpectralClustering(
        n_clusters=optimal_k,
        random_state=config["random_state"],
        affinity="rbf",
        assign_labels="kmeans",
    )
    labels = spectral.fit_predict(features)

    details = {
        "n_clusters": optimal_k,
        "affinity": "rbf",
        "assign_labels": "kmeans",
        "algorithm": "Spectral",
    }

    return labels, details
