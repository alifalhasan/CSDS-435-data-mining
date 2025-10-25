# Project 2: Clustering Algorithms

Implemented and compared three clustering algorithms on gene expression datasets with automatic parameter optimization, visualization, and evaluation.

## Features

- **Three Clustering Algorithms:**
  - K-Means with automatic k selection
  - Hierarchical Clustering (Ward & Complete linkage)
  - DBSCAN with automatic parameter tuning

- **Automatic Parameter Optimization:**
  - Silhouette analysis for optimal k
  - K-distance graph for DBSCAN epsilon
  - Grid search for best parameters

- **Comprehensive Evaluation:**
  - Adjusted Rand Index (ARI) - External Index
  - Jaccard Coefficient - External Index
  - Silhouette Score - Internal Index
  - Cluster count and noise detection

- **Visualizations:**
  - PCA 2D projections (linear dimensionality reduction)
  - t-SNE embeddings (non-linear dimensionality reduction)

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
│   └── dbscan.py              # DBSCAN clustering
├── evaluation/
│   └── metrics.py             # Evaluation metrics
├── visualization/
│   ├── pca_visualizer.py      # PCA visualization
│   └── tsne_visualizer.py     # t-SNE visualization
├── main.py                    # Main script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone or download the project:**
   ```bash
   cd project-directory
   ```

2. **Install dependencies:**
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

This will:
- Load both datasets (cho.txt and iyer.txt)
- Run all three clustering algorithms
- Generate PCA and t-SNE visualizations
- Save numerical results
- Display summary comparison tables

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
    'tsne_perplexity': 30,           # t-SNE perplexity parameter
    'tsne_max_iter': 1000,           # t-SNE maximum iterations
    'plot_dpi': 300,                 # Output image resolution
    'results_dir': 'results',        # Output directory
    'data_dir': 'data'               # Input data directory
}
```

## Output

The script generates the following outputs in the `results/` directory:

### Directory Structure
```
results/
├── pca/
│   ├── cho_pca_comparison.png
│   └── iyer_pca_comparison.png
└── tsne/
    ├── cho_tsne_comparison.png
    └── iyer_tsne_comparison.png
```

### Visualizations
- **PCA plots**: Linear 2D projections showing cluster separation with variance explained
- **t-SNE plots**: Non-linear embeddings revealing complex cluster structures
- Each plot shows true labels and results from all three algorithms with different colors for each cluster

## Performance Metrics

### External Indices (Compare with Ground Truth)

#### Adjusted Rand Index (ARI)
- Measures similarity between predicted and true clusters
- Range: [-1, 1], where 1 = perfect match
- Corrected for chance grouping
- Higher is better

#### Jaccard Coefficient
- Measures agreement between predicted and true clusters using pairwise approach
- Range: [0, 1], where 1 = perfect match
- Counts pairs of samples clustered together correctly
- Higher is better

### Internal Index (No Ground Truth Needed)

#### Silhouette Score
- Measures cluster cohesion and separation
- Range: [-1, 1], where 1 = well-separated clusters
- Evaluates how similar samples are within clusters vs. between clusters
- Higher is better

## Algorithm Details

### K-Means (Partition-Based Clustering)
**Description:** Partitions data into k clusters by minimizing within-cluster variance.

**Features:**
- Automatic k selection via silhouette analysis
- Multiple random initializations to avoid local optima
- Reports inertia (within-cluster sum of squares)

**Pros:**
- Fast and scalable
- Simple to understand and implement
- Works well with spherical clusters

**Cons:**
- Requires specifying number of clusters (k)
- Sensitive to initial centroid placement
- Assumes clusters are spherical and similarly sized
- Sensitive to outliers

### Hierarchical Clustering (Agglomerative)
**Description:** Builds a hierarchy of clusters by iteratively merging the closest pairs.

**Features:**
- Implemented with two inter-cluster distance measures:
  - **Ward linkage**: Minimizes within-cluster variance
  - **Complete linkage**: Uses maximum distance between cluster points
- Uses true cluster count if available, otherwise uses silhouette analysis
- Bottom-up (agglomerative) approach

**Pros:**
- No need to specify number of clusters beforehand
- Creates a dendrogram showing cluster hierarchy
- Deterministic results
- Works with any distance metric

**Cons:**
- Computationally expensive for large datasets (O(n³) time, O(n²) space)
- Once merged, clusters cannot be split
- Sensitive to noise and outliers
- Different linkage methods can produce very different results

### DBSCAN (Density-Based Clustering)
**Description:** Groups together points that are closely packed, marking points in low-density regions as outliers.

**Features:**
- Automatic parameter selection via grid search and k-distance graph
- Tests multiple eps (neighborhood radius) and min_samples combinations
- Selects parameters based on ARI with ground truth
- Automatic fallback using k-distance graph if grid search fails
- Identifies and labels noise points as -1

**Pros:**
- No need to specify number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers (marks them as noise)
- Only two parameters to tune

**Cons:**
- Struggles with varying density clusters
- Sensitive to parameter selection (eps and min_samples)
- Not suitable for high-dimensional data
- Cannot cluster data sets with large differences in densities

## Dependencies

- **numpy**: Numerical computations and array operations
- **pandas**: Data loading and manipulation
- **matplotlib**: Visualization and plotting
- **scikit-learn**: Clustering algorithms, metrics, PCA, and preprocessing

## Notes

- Results directory is automatically cleared on each run to avoid mixing old and new results
- Random seed ensures reproducible results across multiple runs
- All algorithms handle missing values automatically through imputation
- Outliers (label = -1) are excluded from external index calculations (ARI, Jaccard)
- Noise points detected by DBSCAN are excluded from silhouette score calculation

## Visualization Methods

### PCA (Principal Component Analysis)
- **Type**: Linear dimensionality reduction
- **Purpose**: Projects high-dimensional gene expression data into 2D space
- **Advantage**: Preserves global structure, shows variance explained
- **Best for**: Understanding overall data distribution and cluster separation

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Type**: Non-linear dimensionality reduction
- **Purpose**: Reveals complex, non-linear structures in data
- **Advantage**: Better at preserving local structure and revealing tight clusters
- **Best for**: Visualizing fine-grained cluster patterns
- **Note**: Computationally intensive, may take longer to generate