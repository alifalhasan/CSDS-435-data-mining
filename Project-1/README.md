# Project 1: Classification Algorithms

## Project Overview
This project implements and compares 6 classification algorithms on two datasets using 10-fold cross validation.

## Algorithms
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Support Vector Machine (SVM)
- AdaBoost
- Neural Network

## File Structure
```
project1/
├── README.md
├── proj1_description.pdf
├── main.py
├── pytorch_neural_network.py
├── requirements.txt
├── algorithms/
│   ├── adaboost.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── neural_network.py
│   └── svm.py
├── data/
│   ├── project1_dataset1.txt
│   └── project1_dataset2.txt
└── utils/
    ├── data_loader.py
    └── evaluator.py
```

## Installation
1. Download and extract the project folder
2. Install Python (Used version for this project: 3.12.11)
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Used Packages and Versions
- scikit-learn==1.7.2
- pandas==2.3.3
- numpy==2.3.3
- torch==2.8.0

## Usage
Run all algorithms:
```bash
python main.py
```

## Dataset Format
- Place datasets in `data/` folder
- Last column: binary class label
- Other columns: feature values
- Automatic preprocessing included

## Output
- Accuracy, Precision, Recall, F1-Score for each algorithm
- 10-fold cross-validation results
- Performance comparison
- Best algorithm identification

## Algorithm Files
1. `algorithms/knn.py` - KNN
2. `algorithms/decision_tree.py` - Decision Tree
3. `algorithms/naive_bayes.py` - Naive Bayes
4. `algorithms/svm.py` - SVM
5. `algorithms/adaboost.py` - AdaBoost
6. `algorithms/neural_network.py` - Neural Network (sklearn)