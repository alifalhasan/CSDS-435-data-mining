# Clustering Algorithms Comparison Project

A comprehensive framework for comparing multiple clustering algorithms on gene expression datasets with automatic parameter optimization, visualization, and evaluation.

## Features

- **Multiple Clustering Algorithms:**
  - K-Means with automatic k selection
  - Hierarchical Clustering (Ward & Complete linkage)
  - DBSCAN with automatic parameter tuning
  - Gaussian Mixture Model
  - Spectral Clustering

- **Automatic Parameter Optimization:**
  - Silhouette analysis for optimal k
  - K-distance graph for DBSCAN epsilon
  - Grid search for best parameters

- **Comprehensive Evaluation:**
  - Adjusted Rand Index (ARI)
  - Silhouette Score
  - Cluster count and noise detection

- **Rich Visualizations:**
  - PCA 2D projections
  - t-SNE embeddings
  - Side-by-side algorithm comparisons

- **Detailed Reporting:**
  - Console output with progress
  - Text files with numerical results
  - Summary comparison tables

## Project Structure

```
project/
├── utils/
│   ├── config.py              # Configuration settings
│   └── helpers.py             # Helper functions
├── data/
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── cho.txt                # Dataset 1
│   └── iyer.txt               # Dataset 2
├── algorithms/
│   ├── kmeans.py              # K-Means implementation
│   ├── hierarchical.py        # Hierarchical clustering
│   ├── dbscan.py              # DBSCAN clustering
│   ├── gaussian_mixture.py    # Gaussian Mixture Model
│   └── spectral.py            # Spectral clustering
├── evaluation/
│   └── metrics.py             # Evaluation metrics
├── visualization/
│   ├── pca_visualizer.py      # PCA visualization
│   └── tsne_visualizer.py     # t-SNE visualization
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone or download the project:**
   ```bash
   cd project-directory
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Format

The project expects tab-separated text files in the `data/` directory with the following format:

```
gene_id  true_label  feature_1  feature_2  ...  feature_n
gene1    0           0.5        1.2        ...  0.8
gene2    1           1.1        0.3        ...  1.5
...
```

- Column 1: Gene identifier
- Column 2: True cluster label (-1 for outliers)
- Columns 3+: Gene expression features

## Usage

### Basic Usage

Simply run the main script:

```bash
python main.py
```

### Configuration

Edit `utils/config.py` to customize parameters:

```python
CONFIG = {
    'random_state': 42,              # Random seed for reproducibility
    'kmeans_n_init': 10,             # K-Means initializations
    'kmeans_max_iter': 300,          # K-Means max iterations
    'dbscan_eps_options': [...],     # DBSCAN epsilon values to try
    'dbscan_min_samples_options': [...],  # DBSCAN min_samples to try
    'max_k_search': 15,              # Maximum k for silhouette analysis
    'pca_components': 2,             # PCA dimensions
    'tsne_perplexity': 30,           # t-SNE perplexity
    'plot_dpi': 300,                 # Output image resolution
    'results_dir': 'results',        # Output directory
    'data_dir': 'data'               # Input data directory
}
```

### Adding New Datasets

1. Place your `.txt` file in the `data/` directory
2. Update the `dataset_files` list in `main.py`:
   ```python
   dataset_files = ['cho.txt', 'iyer.txt', 'your_dataset.txt']
   ```

### Adding New Algorithms

1. Create a new file in `algorithms/` (e.g., `my_algorithm.py`)
2. Implement your clustering function:
   ```python
   def my_clustering(features, config, true_labels=None):
       # Your implementation
       labels = ...
       details = {'algorithm': 'MyAlgorithm', ...}
       return labels, details
   ```
3. Import and call it in `main.py`

## Output

The script generates the following outputs in the `results/` directory:

### Directory Structure
```
results/
├── pca/
│   ├── cho_pca_comparison.png
│   └── iyer_pca_comparison.png
├── tsne/
│   ├── cho_tsne_comparison.png
│   └── iyer_tsne_comparison.png
└── numerical/
    ├── cho_detailed_results.txt
    ├── iyer_detailed_results.txt
    └── summary_results.txt
```

### Visualizations
- **PCA plots**: 2D projections showing cluster separation
- **t-SNE plots**: Non-linear embeddings revealing cluster structure
- Each plot shows true labels and results from all algorithms

### Numerical Results
- **Detailed files**: Per-dataset metrics for each algorithm
- **Summary file**: Cross-dataset comparison table

### Console Output
- Real-time progress updates
- Dataset statistics
- Algorithm-specific details
- Performance metrics

## Performance Metrics

### Adjusted Rand Index (ARI)
- Measures similarity between predicted and true clusters
- Range: [-1, 1], where 1 = perfect match
- Accounts for chance grouping

### Silhouette Score
- Measures cluster cohesion and separation
- Range: [-1, 1], where 1 = well-separated clusters
- Higher is better

## Algorithm Details

### K-Means
- Automatic k selection via silhouette analysis
- Multiple random initializations
- Reports inertia (within-cluster sum of squares)

### Hierarchical Clustering
- Ward linkage: minimizes variance
- Complete linkage: minimizes maximum distance
- Uses true cluster count if available

### DBSCAN
- Grid search over eps and min_samples
- Automatic fallback using k-distance graph
- Handles noise points (label = -1)

### Gaussian Mixture Model
- Tests multiple covariance types
- Selects best based on log-likelihood
- Probabilistic cluster assignment

### Spectral Clustering
- RBF (Gaussian) affinity matrix
- K-Means for label assignment
- Effective for non-convex clusters

## Troubleshooting

### Memory Issues
- Reduce `max_k_search` in config
- Use smaller datasets
- Limit DBSCAN parameter search space

### Slow t-SNE
- Reduce `tsne_perplexity`
- Reduce `tsne_max_iter`
- Consider using PCA pre-processing

### Import Errors
- Ensure all files are in correct directories
- Check that `__init__.py` files exist if needed
- Verify Python path includes project root

### No Results Generated
- Check `data/` directory contains valid files
- Verify file format matches expected structure
- Check write permissions for `results/` directory

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data loading and manipulation
- **matplotlib**: Visualization
- **scikit-learn**: Machine learning algorithms and metrics
- **scipy**: Scientific computing utilities

## Notes

- Results directory is automatically cleared on each run
- Random seed ensures reproducible results
- All algorithms handle missing values automatically
- Outliers (label = -1) are excluded from ARI calculation