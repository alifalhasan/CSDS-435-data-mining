import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)

datasets = {
    "Dataset1": "data/project1_dataset1.txt",
    "Dataset2": "data/project1_dataset2.txt",
}

######################### asl139 starts ########################

from asl139.decision_tree import run_decision_tree_analysis
from asl139.knn import run_knn_analysis
from asl139.naive_bayes import run_naive_bayes_analysis


def asl139():

    print("\n\n========================================")
    print("Running Decision Tree")
    print("========================================")
    run_decision_tree_analysis(datasets, random_state=42)

    print("\n\n========================================")
    print("Running K-Nearest Neighbors")
    print("========================================")
    run_knn_analysis(datasets, random_state=42)

    print("\n\n========================================")
    print("Running Naive Bayes")
    print("========================================")
    run_naive_bayes_analysis(datasets, random_state=42)


######################### asl139 ends   ########################

from axh1218.helpers.data_loader import DataLoader
from axh1218.helpers.evaluator import Evaluator

from axh1218.algorithms.svm import SVM
from axh1218.algorithms.adaboost import AdaBoost
from axh1218.algorithms.neural_network import NeuralNetwork


def main():

    # Call Alex's parts
    asl139()

    print("\n\n========================================")
    print("axh1218 starts here")
    print("========================================")

    data_loader = DataLoader(random_state=42)
    evaluator = Evaluator(random_state=42)

    models = {
        "Support Vector Machine": SVM(C=1.0, random_state=42),
        "AdaBoost": AdaBoost(n_estimators=4, random_state=42),
        "Neural Network": NeuralNetwork(hidden_layer_sizes=(100, 50), random_state=42),
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
