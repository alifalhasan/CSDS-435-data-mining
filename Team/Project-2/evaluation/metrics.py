import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


def calculate_jaccard_score(true_labels, pred_labels):
    try:
        n = len(true_labels)
        correct = 0
        false_neg = 0
        false_pos = 0

        for i in range(n):
            for j in range(i + 1, n):
                same_true = true_labels[i] == true_labels[j]
                same_pred = pred_labels[i] == pred_labels[j]

                if same_true and same_pred:
                    correct += 1
                elif same_true and not same_pred:
                    false_neg += 1
                elif not same_true and same_pred:
                    false_pos += 1

        # Jaccard coefficient
        if (correct + false_neg + false_pos) > 0:
            return correct / (correct + false_neg + false_pos)
        else:
            return 0.0
    except:
        return -1


def calculate_metrics(true_labels, pred_labels, features):
    # Calculate ARI and Jaccard Coefficient
    mask = true_labels != -1
    if np.sum(mask) > 0 and len(np.unique(pred_labels[mask])) > 1:
        ari = adjusted_rand_score(true_labels[mask], pred_labels[mask])
        jaccard = calculate_jaccard_score(true_labels[mask], pred_labels[mask])
    else:
        ari = -1
        jaccard = -1

    # Calculate silhouette score
    non_noise_mask = pred_labels != -1
    if np.sum(non_noise_mask) > 1 and len(np.unique(pred_labels[non_noise_mask])) > 1:
        try:
            cluster_counts = np.bincount(pred_labels[non_noise_mask])
            if np.all(cluster_counts >= 2):
                silhouette = silhouette_score(
                    features[non_noise_mask], pred_labels[non_noise_mask]
                )
            else:
                silhouette = -1
        except:
            silhouette = -1
    else:
        silhouette = -1

    # Count clusters and noise
    unique_clusters = np.unique(pred_labels)
    n_clusters = (
        len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1
    )
    n_noise = np.sum(pred_labels == -1)

    return {
        "ari": ari,
        "jaccard": jaccard,
        "silhouette": silhouette,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }
