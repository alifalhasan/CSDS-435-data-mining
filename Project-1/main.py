from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

from algorithms.knn import KNN

def main():

    data_loader = DataLoader()
    evaluator = Evaluator()

    datasets = {
        'Dataset1': 'data/project1_dataset1.txt',
        'Dataset2': 'data/project1_dataset2.txt'
    }

    # KNN model
    knn_model = KNN(n_neighbors = 7)

    results = {}

    for name, path in datasets.items():
        print(f"\n--- Processing {name} ---")

        try:
            # Load data
            X, y = data_loader.load_data(path)
            print(f"Number of samples: {X.shape[0]}, Number of input features: {X.shape[1]}")

            scores = evaluator.cross_validate(knn_model, X, y)
            results[name] = scores

            # Print Results
            print("Results:")
            print(f"  Accuracy: {scores['accuracy']:.4f}")
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall: {scores['recall']:.4f}")
            print(f"  F1-Score: {scores['f1_score']:.4f}")

        except Exception as e:
            print(f"Error: {e}")

    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()