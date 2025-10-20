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
    print(f"\n{'='*100}")
    print(f"DETAILED RESULTS - {dataset_name.upper()}")
    print(f"{'='*100}")
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


def print_summary_table(dataset_results_all):
    print(f"\n{'='*120}")
    print(f"SUMMARY COMPARISON - ALL DATASETS AND ALGORITHMS")
    print(f"{'='*120}")

    # Get all dataset names dynamically
    dataset_names = list(dataset_results_all.keys())

    # Header for ARI
    print(f"\n*** ADJUSTED RAND INDEX (ARI) - External Index ***")
    header = f"{'Algorithm':<25}"
    for dataset_name in dataset_names:
        header += f" {dataset_name + ' ARI':<12}"
    header += f" {'Average ARI':<12}"
    print(header)
    print("-" * 120)

    # Get all algorithm names
    all_algorithms = set()
    for results in dataset_results_all.values():
        all_algorithms.update(results.keys())

    # Print each algorithm's ARI results
    for algo_name in sorted(all_algorithms):
        row = f"{algo_name:<25}"
        ari_values = []

        for dataset_name in dataset_names:
            ari = (
                dataset_results_all[dataset_name]
                .get(algo_name, {})
                .get("metrics", {})
                .get("ari", -1)
            )
            if ari != -1:
                row += f" {ari:<12.4f}"
                ari_values.append(ari)
            else:
                row += f" {'N/A':<12}"

        avg_ari = sum(ari_values) / len(ari_values) if ari_values else -1
        row += f" {avg_ari:<12.4f}" if avg_ari != -1 else f" {'N/A':<12}"
        print(row)

    # Header for Jaccard
    print(f"\n*** JACCARD COEFFICIENT - External Index ***")
    header = f"{'Algorithm':<25}"
    for dataset_name in dataset_names:
        header += f" {dataset_name + ' Jaccard':<15}"
    header += f" {'Average Jaccard':<15}"
    print(header)
    print("-" * 120)

    # Print each algorithm's Jaccard results
    for algo_name in sorted(all_algorithms):
        row = f"{algo_name:<25}"
        jaccard_values = []

        for dataset_name in dataset_names:
            jaccard = (
                dataset_results_all[dataset_name]
                .get(algo_name, {})
                .get("metrics", {})
                .get("jaccard", -1)
            )
            if jaccard != -1:
                row += f" {jaccard:<15.4f}"
                jaccard_values.append(jaccard)
            else:
                row += f" {'N/A':<15}"

        avg_jaccard = (
            sum(jaccard_values) / len(jaccard_values) if jaccard_values else -1
        )
        row += f" {avg_jaccard:<15.4f}" if avg_jaccard != -1 else f" {'N/A':<15}"
        print(row)

    # Header for Silhouette (Internal Index)
    print(f"\n*** SILHOUETTE SCORE - Internal Index (for reference) ***")
    header = f"{'Algorithm':<25}"
    for dataset_name in dataset_names:
        header += f" {dataset_name + ' Silh':<15}"
    header += f" {'Average Silh':<15}"
    print(header)
    print("-" * 120)

    # Print each algorithm's Silhouette results
    for algo_name in sorted(all_algorithms):
        row = f"{algo_name:<25}"
        silh_values = []

        for dataset_name in dataset_names:
            silh = (
                dataset_results_all[dataset_name]
                .get(algo_name, {})
                .get("metrics", {})
                .get("silhouette", -1)
            )
            if silh != -1:
                row += f" {silh:<15.4f}"
                silh_values.append(silh)
            else:
                row += f" {'N/A':<15}"

        avg_silh = sum(silh_values) / len(silh_values) if silh_values else -1
        row += f" {avg_silh:<15.4f}" if avg_silh != -1 else f" {'N/A':<15}"
        print(row)


def save_numerical_results(dataset_results_all, results_dir):
    """Save numerical results to files"""
    import os

    # Save detailed results for each dataset
    for dataset_name, results in dataset_results_all.items():
        results_file = os.path.join(
            results_dir, "numerical", f"{dataset_name}_detailed_results.txt"
        )

        with open(results_file, "w") as f:
            f.write(f"CLUSTERING RESULTS - {dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")

            for algo_name, algo_results in results.items():
                metrics = algo_results["metrics"]
                details = algo_results["details"]

                f.write(f"{algo_name}:\n")
                f.write(f"  External Indices:\n")
                f.write(f"    Adjusted Rand Index (ARI): {metrics['ari']:.4f}\n")
                f.write(f"    Jaccard Coefficient: {metrics['jaccard']:.4f}\n")
                f.write(f"  Internal Index:\n")
                f.write(f"    Silhouette Score: {metrics['silhouette']:.4f}\n")
                f.write(f"  Cluster Statistics:\n")
                f.write(f"    Number of Clusters: {metrics['n_clusters']}\n")
                f.write(f"    Number of Noise Points: {metrics['n_noise']}\n")
                f.write(f"  Algorithm Parameters:\n")

                if details["algorithm"] == "KMeans":
                    f.write(f"    Optimal k: {details['n_clusters']}\n")
                    f.write(f"    Inertia: {details['inertia']:.2f}\n")
                elif details["algorithm"] == "Hierarchical":
                    f.write(f"    Number of clusters: {details['n_clusters']}\n")
                    f.write(f"    Linkage: {details['linkage']}\n")
                elif details["algorithm"] == "DBSCAN":
                    f.write(f"    EPS: {details['eps']:.2f}\n")
                    f.write(f"    Min Samples: {details['min_samples']}\n")

                f.write("\n")

    # Save summary results
    summary_file = os.path.join(results_dir, "numerical", "summary_results.txt")
    dataset_names = list(dataset_results_all.keys())

    with open(summary_file, "w") as f:
        f.write("SUMMARY COMPARISON - ALL DATASETS AND ALGORITHMS\n")
        f.write("=" * 100 + "\n\n")

        # Get all algorithms
        all_algorithms = set()
        for results in dataset_results_all.values():
            all_algorithms.update(results.keys())

        # Write ARI table
        f.write("ADJUSTED RAND INDEX (ARI) - External Index\n")
        f.write("-" * 100 + "\n")
        header = f"{'Algorithm':<25}"
        for dataset_name in dataset_names:
            header += f" {dataset_name + ' ARI':<12}"
        header += f" {'Average ARI':<12}\n"
        f.write(header)
        f.write("-" * 100 + "\n")

        for algo_name in sorted(all_algorithms):
            row = f"{algo_name:<25}"
            ari_values = []

            for dataset_name in dataset_names:
                ari = (
                    dataset_results_all[dataset_name]
                    .get(algo_name, {})
                    .get("metrics", {})
                    .get("ari", -1)
                )
                if ari != -1:
                    row += f" {ari:<12.4f}"
                    ari_values.append(ari)
                else:
                    row += f" {'N/A':<12}"

            avg_ari = sum(ari_values) / len(ari_values) if ari_values else -1
            row += f" {avg_ari:<12.4f}" if avg_ari != -1 else f" {'N/A':<12}"
            f.write(row + "\n")

        # Write Jaccard table
        f.write("\n\nJACCARD COEFFICIENT - External Index\n")
        f.write("-" * 100 + "\n")
        header = f"{'Algorithm':<25}"
        for dataset_name in dataset_names:
            header += f" {dataset_name + ' Jaccard':<15}"
        header += f" {'Average Jaccard':<15}\n"
        f.write(header)
        f.write("-" * 100 + "\n")

        for algo_name in sorted(all_algorithms):
            row = f"{algo_name:<25}"
            jaccard_values = []

            for dataset_name in dataset_names:
                jaccard = (
                    dataset_results_all[dataset_name]
                    .get(algo_name, {})
                    .get("metrics", {})
                    .get("jaccard", -1)
                )
                if jaccard != -1:
                    row += f" {jaccard:<15.4f}"
                    jaccard_values.append(jaccard)
                else:
                    row += f" {'N/A':<15}"

            avg_jaccard = (
                sum(jaccard_values) / len(jaccard_values) if jaccard_values else -1
            )
            row += f" {avg_jaccard:<15.4f}" if avg_jaccard != -1 else f" {'N/A':<15}"
            f.write(row + "\n")

        # Write Silhouette table
        f.write("\n\nSILHOUETTE SCORE - Internal Index (for reference)\n")
        f.write("-" * 100 + "\n")
        header = f"{'Algorithm':<25}"
        for dataset_name in dataset_names:
            header += f" {dataset_name + ' Silh':<15}"
        header += f" {'Average Silh':<15}\n"
        f.write(header)
        f.write("-" * 100 + "\n")

        for algo_name in sorted(all_algorithms):
            row = f"{algo_name:<25}"
            silh_values = []

            for dataset_name in dataset_names:
                silh = (
                    dataset_results_all[dataset_name]
                    .get(algo_name, {})
                    .get("metrics", {})
                    .get("silhouette", -1)
                )
                if silh != -1:
                    row += f" {silh:<15.4f}"
                    silh_values.append(silh)
                else:
                    row += f" {'N/A':<15}"

            avg_silh = sum(silh_values) / len(silh_values) if silh_values else -1
            row += f" {avg_silh:<15.4f}" if avg_silh != -1 else f" {'N/A':<15}"
            f.write(row + "\n")


def main():
    print("=" * 100)
    print("CLUSTERING ALGORITHMS")
    print("=" * 100)

    # Setup
    set_random_seed(CONFIG["random_state"])
    results_dir = create_results_folder(CONFIG["results_dir"])

    # Define datasets
    dataset_files = ["cho.txt", "iyer.txt"]
    all_dataset_results = {}

    # Process each dataset
    for dataset_file in dataset_files:
        print(f"\n{'='*80}")
        print(f"ANALYZING DATASET: {dataset_file}")
        print(f"{'='*80}")

        # Load data
        data_dict = load_and_preprocess(dataset_file, CONFIG["data_dir"])
        if data_dict is None:
            continue

        features = data_dict["features"]
        true_labels = data_dict["true_labels"]
        dataset_name = data_dict["dataset_name"]

        dataset_results = {}

        # Run K-Means
        print(f"\n--- KMeans ---")
        try:
            labels, details = kmeans_clustering(features, CONFIG, true_labels)
            metrics = calculate_metrics(true_labels, labels, features)
            dataset_results["KMeans"] = {
                "labels": labels,
                "metrics": metrics,
                "details": details,
            }
            print(
                f"  ARI: {metrics['ari']:.4f}, Jaccard: {metrics['jaccard']:.4f}, Silhouette: {metrics['silhouette']:.4f}, Clusters: {metrics['n_clusters']}"
            )
        except Exception as e:
            print(f"  Error: {e}")

        # Run Hierarchical Ward
        print(f"\n--- Hierarchical_Ward ---")
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
            print(
                f"  ARI: {metrics['ari']:.4f}, Jaccard: {metrics['jaccard']:.4f}, Silhouette: {metrics['silhouette']:.4f}, Clusters: {metrics['n_clusters']}"
            )
        except Exception as e:
            print(f"  Error: {e}")

        # Run Hierarchical Complete
        print(f"\n--- Hierarchical_Complete ---")
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
            print(
                f"  ARI: {metrics['ari']:.4f}, Jaccard: {metrics['jaccard']:.4f}, Silhouette: {metrics['silhouette']:.4f}, Clusters: {metrics['n_clusters']}"
            )
        except Exception as e:
            print(f"  Error: {e}")

        # Run DBSCAN
        print(f"\n--- DBSCAN ---")
        try:
            labels, details = dbscan_clustering(features, CONFIG, true_labels)
            metrics = calculate_metrics(true_labels, labels, features)
            dataset_results["DBSCAN"] = {
                "labels": labels,
                "metrics": metrics,
                "details": details,
            }
            print(
                f"  ARI: {metrics['ari']:.4f}, Jaccard: {metrics['jaccard']:.4f}, Silhouette: {metrics['silhouette']:.4f}, Clusters: {metrics['n_clusters']}"
            )
        except Exception as e:
            print(f"  Error: {e}")

        # Store results
        all_dataset_results[dataset_name] = dataset_results

        # Print detailed results
        print_detailed_results(dataset_name, dataset_results)

        # Create visualizations
        print(f"\nCreating visualizations for {dataset_name}...")

        print("Creating PCA visualization...")
        create_pca_comparison(
            features, true_labels, dataset_results, dataset_name, CONFIG, results_dir
        )

        print("Creating t-SNE visualization...")
        create_tsne_comparison(
            features, true_labels, dataset_results, dataset_name, CONFIG, results_dir
        )

    # Save numerical results
    print(f"\nSaving numerical results to {results_dir}...")
    save_numerical_results(all_dataset_results, results_dir)

    # Print final summary
    print_summary_table(all_dataset_results)

    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE!")
    print(f"All results saved in: {results_dir}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
