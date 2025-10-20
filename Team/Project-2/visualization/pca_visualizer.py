import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def create_pca_comparison(
    features, true_labels, algorithm_results, dataset_name, config, results_dir
):
    pca = PCA(
        n_components=config["pca_components"], random_state=config["random_state"]
    )
    features_pca = pca.fit_transform(features)
    variance_explained = pca.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(2, 3, figsize=config["figure_size"])
    axes = axes.flatten()

    # Plot true labels
    scatter = axes[0].scatter(
        features_pca[:, 0],
        features_pca[:, 1],
        c=true_labels,
        cmap="tab20",
        alpha=0.7,
        s=30,
    )
    axes[0].set_title(f"True Labels\nVariance: {variance_explained:.3f}", fontsize=10)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(scatter, ax=axes[0])

    # Plot algorithms
    for idx, (algo_name, results) in enumerate(algorithm_results.items(), 1):
        if idx < len(axes):
            ari = results["metrics"]["ari"]
            silhouette = results["metrics"]["silhouette"]
            scatter = axes[idx].scatter(
                features_pca[:, 0],
                features_pca[:, 1],
                c=results["labels"],
                cmap="tab20",
                alpha=0.7,
                s=30,
            )
            axes[idx].set_title(
                f"{algo_name}\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}",
                fontsize=10,
            )
            axes[idx].set_xlabel("PC1")
            axes[idx].set_ylabel("PC2")
            plt.colorbar(scatter, ax=axes[idx])

    # Hide empty subplots
    for idx in range(len(algorithm_results) + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"PCA Visualization - {dataset_name}", fontsize=16, y=0.95)
    plt.tight_layout()

    filename = os.path.join(results_dir, "pca", f"{dataset_name}_pca_comparison.png")
    plt.savefig(filename, dpi=config["plot_dpi"], bbox_inches="tight")
    print(f"Saved PCA visualization: {filename}")
    plt.show()
