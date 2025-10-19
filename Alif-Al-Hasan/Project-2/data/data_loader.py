import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_dataset(self, filename):
        """Load gene expression dataset"""
        try:
            data = pd.read_csv(f'data/{filename}', sep='\t', header=None)
            gene_ids = data.iloc[:, 0].values
            true_labels = data.iloc[:, 1].values
            features = data.iloc[:, 2:].values
            
            print(f"Loaded {filename}:")
            print(f"  - Samples: {len(gene_ids)}")
            print(f"  - Features: {features.shape[1]}")
            print(f"  - True clusters: {len(np.unique(true_labels[true_labels != -1]))}")
            print(f"  - Outliers: {np.sum(true_labels == -1)}")
            
            return gene_ids, true_labels, features
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None, None
    
    def preprocess_features(self, features):
        """Preprocess the feature data"""
        # Remove columns with all zeros or constant values
        non_constant_cols = ~(np.all(features == features[0, :], axis=0) | 
                             np.all(features == 0, axis=0))
        features = features[:, non_constant_cols]
        
        # Handle missing values
        features = self.imputer.fit_transform(features)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
    
    def load_and_preprocess(self, filename):
        """Complete data loading and preprocessing pipeline"""
        gene_ids, true_labels, features = self.load_dataset(filename)
        
        if features is None:
            return None
        
        features_processed = self.preprocess_features(features)
        
        return {
            'gene_ids': gene_ids,
            'true_labels': true_labels,
            'features': features_processed,
            'dataset_name': filename.replace('.txt', '')
        }