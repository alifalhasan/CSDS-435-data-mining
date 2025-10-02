from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

from algorithms.knn import KNN
from algorithms.decision_tree import DecisionTree
from algorithms.naive_bayes import NaiveBayes
from algorithms.svm import SVM


def main():

    data_loader = DataLoader()
    evaluator = Evaluator()

    datasets = {
        "Dataset1": "data/project1_dataset1.txt",
        "Dataset2": "data/project1_dataset2.txt",
    }

    models = {
        "K-Nearest Neighbors": KNN(n_neighbors=7),
        "Decision Tree": DecisionTree(max_depth=5),
        "Naive Bayes": NaiveBayes(),
        "Support Vector Machine": SVM(C=1.0),
    }

    results = {}

    for name, path in datasets.items():
        print(f"\n--- Processing {name} ---")

        try:
            # Load data
            X, y = data_loader.load_data(path)
            print(
                f"Number of samples: {X.shape[0]}, Number of input features: {X.shape[1]}"
            )

            dataset_results = {}

            # Evaluate models
            for model_name, model in models.items():
                print(f"\n--- Evaluating {model_name} ---")
                scores = evaluator.cross_validate(model, X, y)
                dataset_results[model_name] = scores

                # Print Results
                print(f"  Accuracy: {scores['accuracy']:.4f}")
                print(f"  Precision: {scores['precision']:.4f}")
                print(f"  Recall: {scores['recall']:.4f}")
                print(f"  F1-Score: {scores['f1_score']:.4f}")

            results[name] = dataset_results

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
