from sklearn.cluster import KMeans
from utils.helpers import determine_optimal_k


def kmeans_clustering(features, config, true_labels=None):
    optimal_k = determine_optimal_k(features, config["max_k_search"])

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=config["random_state"],
        n_init=config["kmeans_n_init"],
        max_iter=config["kmeans_max_iter"],
    )
    labels = kmeans.fit_predict(features)

    details = {
        "n_clusters": optimal_k,
        "inertia": kmeans.inertia_,
        "algorithm": "KMeans",
    }

    return labels, details
