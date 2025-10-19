import numpy as np
from sklearn.mixture import GaussianMixture
from utils.helpers import determine_optimal_k


def gaussian_mixture_clustering(features, config, true_labels=None):
    optimal_k = determine_optimal_k(features, config["max_k_search"])

    best_labels = None
    best_score = -np.inf
    best_cov_type = "full"

    for cov_type in config["gmm_covariance_types"]:
        try:
            gmm = GaussianMixture(
                n_components=optimal_k,
                covariance_type=cov_type,
                random_state=config["random_state"],
                max_iter=config["gmm_max_iter"],
            )
            labels = gmm.fit_predict(features)
            score = gmm.score(features)

            if score > best_score:
                best_score = score
                best_labels = labels
                best_cov_type = cov_type
        except:
            continue

    if best_labels is None:
        gmm = GaussianMixture(
            n_components=optimal_k, random_state=config["random_state"]
        )
        best_labels = gmm.fit_predict(features)
        best_cov_type = "diag"

    details = {
        "n_clusters": optimal_k,
        "covariance_type": best_cov_type,
        "log_likelihood": best_score,
        "algorithm": "GaussianMixture",
    }

    return best_labels, details
