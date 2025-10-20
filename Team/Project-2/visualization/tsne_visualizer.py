import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def create_tsne_comparison(
    features, true_labels, algorithm_results, dataset_name, config, results_dir
):
    try:
        tsne = TSNE(
            n_components=2,
            random_state=config["random_state"],
            perplexity=config["tsne_perplexity"],
            learning_rate="auto",
            n_iter=config["tsne_max_iter"],
        )
    except TypeError:
        try:
            tsne = TSNE(
                n_components=2,
                random_state=config["random_state"],
                perplexity=config["tsne_perplexity"],
                learning_rate="auto",
                max_iter=config["tsne_max_iter"],
            )
        except TypeError:
            tsne = TSNE(
                n_components=2,
                random_state=config["random_state"],
                perplexity=config["tsne_perplexity"],
            )

    features_tsne = tsne.fit_transform(features)

    fig, axes = plt.subplots(2, 3, figsize=config["figure_size"])
    axes = axes.flatten()

    # Plot true labels
    scatter = axes[0].scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=true_labels,
        cmap="tab20",
        alpha=0.7,
        s=30,
    )
    axes[0].set_title("True Labels", fontsize=10)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[0])

    # Plot algorithms
    for idx, (algo_name, results) in enumerate(algorithm_results.items(), 1):
        if idx < len(axes):
            ari = results["metrics"]["ari"]
            silhouette = results["metrics"]["silhouette"]
            scatter = axes[idx].scatter(
                features_tsne[:, 0],
                features_tsne[:, 1],
                c=results["labels"],
                cmap="tab20",
                alpha=0.7,
                s=30,
            )
            axes[idx].set_title(
                f"{algo_name}\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}",
                fontsize=10,
            )
            axes[idx].set_xlabel("t-SNE 1")
            axes[idx].set_ylabel("t-SNE 2")
            plt.colorbar(scatter, ax=axes[idx])

    # Hide empty subplots
    for idx in range(len(algorithm_results) + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"t-SNE Visualization - {dataset_name}", fontsize=16, y=0.95)
    plt.tight_layout()

    filename = os.path.join(results_dir, "tsne", f"{dataset_name}_tsne_comparison.png")
    plt.savefig(filename, dpi=config["plot_dpi"], bbox_inches="tight")
    plt.show()
