# Configuration settings for the entire project
CONFIG = {
    'random_state': 42,
    'kmeans_n_init': 10,
    'kmeans_max_iter': 300,
    'hierarchical_linkage': ['ward', 'complete'],
    'dbscan_eps_options': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    'dbscan_min_samples_options': [3, 5, 7],
    'gmm_covariance_types': ['spherical', 'tied', 'diag', 'full'],
    'gmm_max_iter': 100,
    'spectral_n_init': 10,
    'pca_components': 2,
    'tsne_perplexity': 30,
    'tsne_learning_rate': 200,
    'tsne_n_iter': 1000,  # This is the correct parameter name
    'plot_dpi': 300,
    'figure_size': (20, 15)
}