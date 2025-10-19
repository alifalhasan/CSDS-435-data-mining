import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add paths to import modules
sys.path.append('./data')
sys.path.append('./algorithms')
sys.path.append('./visualization')
sys.path.append('./evaluation')
sys.path.append('./utils')

from data.data_loader import DataLoader
from algorithms.kmeans import KMeansClustering
from algorithms.hierarchical import HierarchicalClustering
from algorithms.dbscan import DBSCANClustering
from algorithms.gaussian_mixture import GaussianMixtureClustering
from algorithms.spectral import SpectralClustering
from visualization.pca_visualizer import PCAVisualizer
from visualization.tsne_visualizer import TSNEVisualizer
from evaluation.metrics import EvaluationMetrics
from utils.config import CONFIG
from utils.helpers import set_random_seed, create_results_folder

def print_detailed_results(dataset_name, dataset_results):
    """Print detailed numerical results to terminal"""
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS - {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Algorithm':<25} {'ARI':<8} {'Silhouette':<12} {'Clusters':<10} {'Noise':<8} {'Details'}")
    print(f"{'-'*80}")
    
    for algo_name, results in dataset_results.items():
        metrics = results['metrics']
        details = results['details']
        
        # Format details string
        details_str = ""
        if details['algorithm'] == 'KMeans':
            details_str = f"k={details['n_clusters']}, inertia={details['inertia']:.2f}"
        elif details['algorithm'] == 'Hierarchical':
            details_str = f"k={details['n_clusters']}, linkage={details['linkage']}"
        elif details['algorithm'] == 'DBSCAN':
            details_str = f"eps={details['eps']:.2f}, min_samples={details['min_samples']}"
        elif details['algorithm'] == 'GaussianMixture':
            details_str = f"k={details['n_clusters']}, cov_type={details['covariance_type']}"
        elif details['algorithm'] == 'Spectral':
            details_str = f"k={details['n_clusters']}, affinity={details['affinity']}"
        
        print(f"{algo_name:<25} {metrics['ari']:<8.4f} {metrics['silhouette']:<12.4f} "
              f"{metrics['n_clusters']:<10} {metrics['n_noise']:<8} {details_str}")

def print_summary_table(dataset_results_all):
    """Print summary table comparing all datasets and algorithms"""
    print(f"\n{'='*100}")
    print(f"SUMMARY COMPARISON - ALL DATASETS AND ALGORITHMS")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Algorithm':<25}"
    for dataset_name in dataset_results_all.keys():
        header += f" {dataset_name + ' ARI':<12}"
    header += f" {'Average ARI':<12}"
    print(header)
    print('-' * 100)
    
    # Calculate averages
    algorithm_aris = {}
    
    for dataset_name, results in dataset_results_all.items():
        for algo_name, algo_results in results.items():
            if algo_name not in algorithm_aris:
                algorithm_aris[algo_name] = []
            algorithm_aris[algo_name].append(algo_results['metrics']['ari'])
    
    # Print each algorithm's results
    for algo_name in sorted(algorithm_aris.keys()):
        row = f"{algo_name:<25}"
        total_ari = 0
        count = 0
        
        for dataset_name in dataset_results_all.keys():
            ari = dataset_results_all[dataset_name].get(algo_name, {}).get('metrics', {}).get('ari', -1)
            if ari != -1:
                row += f" {ari:<12.4f}"
                total_ari += ari
                count += 1
            else:
                row += f" {'N/A':<12}"
        
        avg_ari = total_ari / count if count > 0 else -1
        row += f" {avg_ari:<12.4f}" if avg_ari != -1 else f" {'N/A':<12}"
        print(row)

def save_numerical_results(dataset_results_all, results_dir):
    """Save all numerical results to text files"""
    # Save detailed results for each dataset
    for dataset_name, results in dataset_results_all.items():
        results_file = os.path.join(results_dir, "numerical", f"{dataset_name}_detailed_results.txt")
        
        with open(results_file, 'w') as f:
            f.write(f"CLUSTERING RESULTS - {dataset_name.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            for algo_name, algo_results in results.items():
                metrics = algo_results['metrics']
                details = algo_results['details']
                
                f.write(f"{algo_name}:\n")
                f.write(f"  Adjusted Rand Index: {metrics['ari']:.4f}\n")
                f.write(f"  Silhouette Score: {metrics['silhouette']:.4f}\n")
                f.write(f"  Number of Clusters: {metrics['n_clusters']}\n")
                f.write(f"  Number of Noise Points: {metrics['n_noise']}\n")
                
                # Algorithm-specific details
                if details['algorithm'] == 'KMeans':
                    f.write(f"  Optimal k: {details['n_clusters']}\n")
                    f.write(f"  Inertia: {details['inertia']:.2f}\n")
                elif details['algorithm'] == 'Hierarchical':
                    f.write(f"  Linkage: {details['linkage']}\n")
                elif details['algorithm'] == 'DBSCAN':
                    f.write(f"  EPS: {details['eps']:.2f}\n")
                    f.write(f"  Min Samples: {details['min_samples']}\n")
                elif details['algorithm'] == 'GaussianMixture':
                    f.write(f"  Covariance Type: {details['covariance_type']}\n")
                elif details['algorithm'] == 'Spectral':
                    f.write(f"  Affinity: {details['affinity']}\n")
                
                f.write("\n")
    
    # Save summary results
    summary_file = os.path.join(results_dir, "numerical", "summary_results.txt")
    with open(summary_file, 'w') as f:
        f.write("SUMMARY COMPARISON - ALL DATASETS AND ALGORITHMS\n")
        f.write("=" * 80 + "\n\n")
        
        # Calculate averages for summary
        algorithm_aris = {}
        for dataset_name, results in dataset_results_all.items():
            for algo_name, algo_results in results.items():
                if algo_name not in algorithm_aris:
                    algorithm_aris[algo_name] = []
                algorithm_aris[algo_name].append(algo_results['metrics']['ari'])
        
        # Write summary table
        f.write(f"{'Algorithm':<25} {'cho ARI':<12} {'iyer ARI':<12} {'Average ARI':<12}\n")
        f.write("-" * 80 + "\n")
        
        for algo_name in sorted(algorithm_aris.keys()):
            cho_ari = dataset_results_all['cho'].get(algo_name, {}).get('metrics', {}).get('ari', -1)
            iyer_ari = dataset_results_all['iyer'].get(algo_name, {}).get('metrics', {}).get('ari', -1)
            
            ari_values = [ari for ari in [cho_ari, iyer_ari] if ari != -1]
            avg_ari = sum(ari_values) / len(ari_values) if ari_values else -1
            
            f.write(f"{algo_name:<25} ")
            f.write(f"{cho_ari:<12.4f}" if cho_ari != -1 else f"{'N/A':<12}")
            f.write(f"{iyer_ari:<12.4f}" if iyer_ari != -1 else f"{'N/A':<12}")
            f.write(f"{avg_ari:<12.4f}" if avg_ari != -1 else f"{'N/A':<12}")
            f.write("\n")

def main():
    """Main function to run complete clustering analysis"""
    print("=" * 100)
    print("CLUSTERING ALGORITHMS COMPARISON PROJECT")
    print("=" * 100)
    
    # Set random seeds for reproducibility
    set_random_seed(CONFIG['random_state'])
    
    # Create results folder (overwrites existing)
    results_dir = create_results_folder()
    print(f"Results will be saved in: {results_dir}")
    
    # Initialize components
    data_loader = DataLoader()
    evaluator = EvaluationMetrics()
    pca_visualizer = PCAVisualizer(results_dir)
    tsne_visualizer = TSNEVisualizer(results_dir)
    
    # Define all clustering algorithms
    clustering_algorithms = {
        'KMeans': KMeansClustering(),
        'Hierarchical_Ward': HierarchicalClustering(linkage='ward'),
        'Hierarchical_Complete': HierarchicalClustering(linkage='complete'),
        'DBSCAN': DBSCANClustering(),
        'GaussianMixture': GaussianMixtureClustering(),
        'Spectral': SpectralClustering()
    }
    
    # Dataset files
    dataset_files = ['cho.txt', 'iyer.txt']
    all_dataset_results = {}
    
    # Run analysis for each dataset
    for dataset_file in dataset_files:
        print(f"\n{'='*80}")
        print(f"ANALYZING DATASET: {dataset_file}")
        print(f"{'='*80}")
        
        # Load and preprocess data
        dataset_name = dataset_file.replace('.txt', '')
        data_dict = data_loader.load_and_preprocess(dataset_file)
        
        if data_dict is None:
            continue
            
        features = data_dict['features']
        true_labels = data_dict['true_labels']
        
        # Store results for this dataset
        dataset_results = {}
        
        # Run each clustering algorithm
        for algo_name, algorithm in clustering_algorithms.items():
            print(f"\n--- {algo_name} ---")
            
            try:
                # Perform clustering
                pred_labels, algo_details = algorithm.fit_predict(
                    features, true_labels, dataset_name
                )
                
                # Evaluate results
                metrics = evaluator.calculate_metrics(true_labels, pred_labels, features)
                
                # Store results
                dataset_results[algo_name] = {
                    'labels': pred_labels,
                    'metrics': metrics,
                    'details': algo_details
                }
                
                print(f"  ARI: {metrics['ari']:.4f}")
                print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
                print(f"  Clusters found: {metrics['n_clusters']}")
                print(f"  Noise points: {metrics['n_noise']}")
                
                # Print algorithm-specific details
                if algo_details['algorithm'] == 'KMeans':
                    print(f"  Optimal k: {algo_details['n_clusters']}")
                    print(f"  Inertia: {algo_details['inertia']:.2f}")
                elif algo_details['algorithm'] == 'Hierarchical':
                    print(f"  Linkage: {algo_details['linkage']}")
                elif algo_details['algorithm'] == 'DBSCAN':
                    print(f"  EPS: {algo_details['eps']:.2f}")
                    print(f"  Min samples: {algo_details['min_samples']}")
                elif algo_details['algorithm'] == 'GaussianMixture':
                    print(f"  Covariance type: {algo_details['covariance_type']}")
                elif algo_details['algorithm'] == 'Spectral':
                    print(f"  Affinity: {algo_details['affinity']}")
                      
            except Exception as e:
                print(f"Error in {algo_name}: {e}")
                continue
        
        # Store dataset results
        all_dataset_results[dataset_name] = dataset_results
        
        # Print detailed results table
        print_detailed_results(dataset_name, dataset_results)
        
        # Create visualizations (both display and save)
        print(f"\nCreating visualizations for {dataset_name}...")
        
        # PCA Visualization
        print("Displaying and saving PCA visualization...")
        pca_visualizer.create_comparison_plots(
            features, true_labels, dataset_results, dataset_name
        )
        
        # t-SNE Visualization
        print("Displaying and saving t-SNE visualization...")
        tsne_visualizer.create_comparison_plots(
            features, true_labels, dataset_results, dataset_name
        )
    
    # Save all numerical results
    print(f"\nSaving numerical results to {results_dir}...")
    save_numerical_results(all_dataset_results, results_dir)
    
    # Print final summary comparison
    print_summary_table(all_dataset_results)
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE!")
    print(f"All results saved in: {results_dir}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()