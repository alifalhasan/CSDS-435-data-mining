import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
from utils.config import CONFIG

class TSNEVisualizer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
    
    def create_comparison_plots(self, features, true_labels, algorithm_results, dataset_name):
        """Create t-SNE comparison plots for all algorithms - display and save"""
        print("Computing t-SNE embedding (this may take a while)...")
        
        # Apply t-SNE with compatibility for different scikit-learn versions
        try:
            # Try newer parameter names first
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=CONFIG['tsne_perplexity'],
                learning_rate=CONFIG['tsne_learning_rate'],
                n_iter=CONFIG['tsne_n_iter']
            )
        except TypeError:
            try:
                # Try with max_iter instead of n_iter
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=CONFIG['tsne_perplexity'],
                    learning_rate=CONFIG['tsne_learning_rate'],
                    max_iter=CONFIG['tsne_n_iter']
                )
            except TypeError:
                # Use minimal parameters for very old versions
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=CONFIG['tsne_perplexity']
                )
        
        features_tsne = tsne.fit_transform(features)
        
        # Create subplots
        n_algorithms = len(algorithm_results)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot true labels
        self._plot_tsne(axes[0], features_tsne, true_labels, 'True Labels')
        
        # Plot each algorithm
        for idx, (algo_name, results) in enumerate(algorithm_results.items(), 1):
            if idx < len(axes):
                ari = results['metrics']['ari']
                silhouette = results['metrics']['silhouette']
                self._plot_tsne(axes[idx], features_tsne, results['labels'],
                               f'{algo_name}\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}')
        
        # Hide empty subplots
        for idx in range(len(algorithm_results) + 1, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f't-SNE Visualization - {dataset_name}', fontsize=16, y=0.95)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, "tsne", f'{dataset_name}_tsne_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization: {filename}")
        
        # Display figure
        plt.show()
    
    def _plot_tsne(self, ax, features_tsne, labels, title):
        """Plot individual t-SNE visualization"""
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                           c=labels, cmap='tab20', alpha=0.7, s=30)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax)