import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

class PCAVisualizer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
    
    def create_comparison_plots(self, features, true_labels, algorithm_results, dataset_name):
        """Create PCA comparison plots for all algorithms - display and save"""
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(features)
        variance_explained = pca.explained_variance_ratio_.sum()
        
        # Create subplots
        n_algorithms = len(algorithm_results)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot true labels
        self._plot_pca(axes[0], features_pca, true_labels, 
                      f'True Labels\nVariance: {variance_explained:.3f}')
        
        # Plot each algorithm
        for idx, (algo_name, results) in enumerate(algorithm_results.items(), 1):
            if idx < len(axes):
                ari = results['metrics']['ari']
                silhouette = results['metrics']['silhouette']
                self._plot_pca(axes[idx], features_pca, results['labels'],
                              f'{algo_name}\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}')
        
        # Hide empty subplots
        for idx in range(len(algorithm_results) + 1, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'PCA Visualization - {dataset_name}', fontsize=16, y=0.95)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, "pca", f'{dataset_name}_pca_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved PCA visualization: {filename}")
        
        # Display figure
        plt.show()
    
    def _plot_pca(self, ax, features_pca, labels, title):
        """Plot individual PCA visualization"""
        scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                           c=labels, cmap='tab20', alpha=0.7, s=30)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax)