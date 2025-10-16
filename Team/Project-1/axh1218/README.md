# Classification Algorithms Project

## Student Information
- **Name**: Alif Al Hasan
- **Case ID**: axh1218
- **Algorithms Implemented**: Support Vector Machine (SVM), AdaBoost, Neural Network

## Folder Structure
```
axh1218/
├── README.md                   # Project documentation
├── algorithms/                 # Main algorithm implementations
│   ├── svm.py                   # Support Vector Machine
│   ├── adaboost.py              # AdaBoost classifier
│   └── neural_network.py        # Neural Network classifier
└── helpers/                    # Supporting utilities
    ├── data_loader.py           # Data loading and preprocessing
    └── evaluator.py             # Model evaluation tools
```

## Algorithms Implemented

### 1. Support Vector Machine (SVM)
**Location**: `algorithms/svm.py`

SVM is a powerful algorithm that finds the best boundary to separate different classes in the data.

**Key Features**:
- Uses RBF (Radial Basis Function) kernel to handle complex, non-linear patterns
- Automatic gamma scaling for optimal performance
- L2 regularization (C=1.0) to prevent the model from becoming too complex

**Best for**: High-dimensional data and complex classification problems

### 2. AdaBoost
**Location**: `algorithms/adaboost.py`

AdaBoost combines multiple simple models to create a stronger, more accurate classifier. It focuses more on examples that are hard to classify correctly.

**Key Features**:
- Uses Decision Trees as building blocks
- 4 boosting rounds to improve accuracy
- SAMME algorithm for better probability estimates

**Best for**: Situations where one wants to boost the performance of simple models

### 3. Neural Network
**Location**: `algorithms/neural_network.py`

A neural network capable of learning complex patterns through multiple layers.

**Key Features**:
- Two hidden layers with 100 and 50 neurons respectively
- ReLU activation functions for better learning
- Adam optimizer that adapts the learning rate automatically
- Early stopping to avoid overfitting

**Best for**: Complex patterns and large datasets

## Helper Modules

### Data Loader (`helpers/data_loader.py`)
Handles all data preparation tasks to ensure the algorithms receive properly formatted data.

**Functions**:
- Loads datasets from files
- Processes both numerical data and categorical data
- Automatically scales features to similar ranges
- Converts categorical values into numbers the algorithms can understand

### Evaluator (`helpers/evaluator.py`)
Measures how well each algorithm performs using standard evaluation methods.

**Functions**:
- Performs 10-fold cross-validation
- Calculates accuracy, precision, recall, and F1-score
- Uses fixed random splits for consistent, reproducible results

## How to Use

### Testing Individual Algorithms
Each algorithm can be tested separately using this simple pattern:

```python
from helpers.data_loader import DataLoader
from helpers.evaluator import Evaluator
from algorithms.svm import SVM

# Step 1: Load your data
data_loader = DataLoader()
X, y = data_loader.load_data('data/dataset.txt')

# Step 2: Choose and initialize an algorithm
model = SVM()

# Step 3: Evaluate the model
evaluator = Evaluator()
scores = evaluator.cross_validate(model, X, y)
```

### Integration with Main Project
All algorithms follow the same standard format:
- Each algorithm has `fit()` method to train the model
- Each algorithm has `predict()` method to make predictions
- Data preprocessing happens automatically
- Results are reproducible every time you run the code

## File Descriptions

### `algorithms/svm.py`
Implements the Support Vector Machine classifier using RBF kernel. This file contains the complete SVM implementation with optimized parameters.

### `algorithms/adaboost.py`
Implements the AdaBoost algorithm with decision tree base learners. Uses boosting to improve classification accuracy by combining weak learners.

### `algorithms/neural_network.py`
Implements a multi-layer neural network using scikit-learn's MLPClassifier. Includes features like early stopping and adaptive learning rate.

### `helpers/data_loader.py`
Contains utilities for loading and preprocessing data. Ensures all algorithms receive properly formatted input data.

### `helpers/evaluator.py`
Provides evaluation framework for measuring and comparing algorithm performance using cross-validation and multiple metrics.

## Evaluation Metrics

The project measures algorithm performance using four key metrics:

1. **Accuracy**: Overall percentage of correct predictions
2. **Precision**: How many predicted positive cases are actually positive
3. **Recall**: How many actual positive cases were correctly identified
4. **F1-Score**: Balance between precision and recall