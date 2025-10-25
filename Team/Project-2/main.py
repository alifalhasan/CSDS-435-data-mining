from utils.config import CONFIG
from utils.helpers import set_random_seed, create_results_folder

from data.data_loader import load_and_preprocess

from algorithms.kmeans import kmeans_clustering
from algorithms.hierarchical import hierarchical_clustering
from algorithms.dbscan import dbscan_clustering

from evaluation.metrics import calculate_metrics

from visualization.pca_visualizer import create_pca_comparison
from visualization.tsne_visualizer import create_tsne_comparison


def print_detailed_results(dataset_name, dataset_results):
    print(f"Results for {dataset_name.upper()} dataset:")
    print()
    print(
        f"{'Algorithm':<25} {'ARI':<10} {'Jaccard':<10} {'Silhouette':<12} {'Clusters':<10} {'Noise':<8} {'Details'}"
    )
    print(f"{'-'*100}")

    for algo_name, results in dataset_results.items():
        metrics = results["metrics"]
        details = results["details"]

        details_str = ""
        if details["algorithm"] == "KMeans":
            details_str = f"k={details['n_clusters']}, inertia={details['inertia']:.2f}"
        elif details["algorithm"] == "Hierarchical":
            details_str = f"k={details['n_clusters']}, linkage={details['linkage']}"
        elif details["algorithm"] == "DBSCAN":
            details_str = (
                f"eps={details['eps']:.2f}, min_samples={details['min_samples']}"
            )

        print(
            f"{algo_name:<25} {metrics['ari']:<10.4f} {metrics['jaccard']:<10.4f} {metrics['silhouette']:<12.4f} "
            f"{metrics['n_clusters']:<10} {metrics['n_noise']:<8} {details_str}"
        )


def main():
    set_random_seed(CONFIG["random_state"])
    results_dir = create_results_folder(CONFIG["results_dir"])

    dataset_files = ["cho.txt", "iyer.txt"]
    all_dataset_results = {}

    for dataset_file in dataset_files:
        data_dict = load_and_preprocess(dataset_file, CONFIG["data_dir"])
        if data_dict is None:
            continue

        features = data_dict["features"]
        true_labels = data_dict["true_labels"]
        dataset_name = data_dict["dataset_name"]

        dataset_results = {}

        # Run K-Means
        try:
            labels, details = kmeans_clustering(features, CONFIG, true_labels)
            metrics = calculate_metrics(true_labels, labels, features)
            dataset_results["KMeans"] = {
                "labels": labels,
                "metrics": metrics,
                "details": details,
            }
        except Exception as e:
            print(f"  Error: {e}")

        # Run Hierarchical Ward
        try:
            labels, details = hierarchical_clustering(
                features, CONFIG, "ward", true_labels
            )
            metrics = calculate_metrics(true_labels, labels, features)
            dataset_results["Hierarchical_Ward"] = {
                "labels": labels,
                "metrics": metrics,
                "details": details,
            }
        except Exception as e:
            print(f"  Error: {e}")

        # Run Hierarchical Complete
        try:
            labels, details = hierarchical_clustering(
                features, CONFIG, "complete", true_labels
            )
            metrics = calculate_metrics(true_labels, labels, features)
            dataset_results["Hierarchical_Complete"] = {
                "labels": labels,
                "metrics": metrics,
                "details": details,
            }
        except Exception as e:
            print(f"  Error: {e}")

        # Run DBSCAN
        try:
            labels, details = dbscan_clustering(features, CONFIG, true_labels)
            metrics = calculate_metrics(true_labels, labels, features)
            dataset_results["DBSCAN"] = {
                "labels": labels,
                "metrics": metrics,
                "details": details,
            }
        except Exception as e:
            print(f"  Error: {e}")

        all_dataset_results[dataset_name] = dataset_results

        print_detailed_results(dataset_name, dataset_results)

        create_pca_comparison(
            features, true_labels, dataset_results, dataset_name, CONFIG, results_dir
        )

        create_tsne_comparison(
            features, true_labels, dataset_results, dataset_name, CONFIG, results_dir
        )


if __name__ == "__main__":
    main()
