# Classification Algorithms Project

## Team Members
- **Alexander Lott** (Case ID: asl139) - KNN, Decision Tree, Naive Bayes
- **Alif Al Hasan** (Case ID: axh1218) - SVM, AdaBoost, Neural Network

## Project Overview
This project implements and compares six classification algorithms on two datasets using 10-fold cross validation.

## Project Structure
```
root/
├── README.md
├── main.py
├── requirements.txt
├── data/
│   ├── project1_dataset1.txt
│   └── project1_dataset2.txt
├── axh1218/
│   ├── README.md
│   ├── algorithms/
│   │   ├── svm.py
│   │   ├── adaboost.py
│   │   └── neural_network.py
│   └── helpers/
│       ├── data_loader.py
│       └── evaluator.py
│── asl139/
│   ├── decision_tree.py
│   ├── knn.py
│   └── naive_bayes.py
```

## Algorithms Implemented

### By Alexander Lott (asl139)
1. **K-Nearest Neighbors (KNN)** - `knn.py`
2. **Decision Tree** - `decision_tree.py`
3. **Naive Bayes** - `naive_bayes.py`

### By Alif Al Hasan (axh1218)
1. **Support Vector Machine (SVM)** - `axh1218/algorithms/svm.py`
2. **AdaBoost** - `axh1218/algorithms/adaboost.py`
3. **Neural Network** - `axh1218/algorithms/neural_network.py`

## Installation

1. **Clone or download** the project folder
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Required Packages (Used version)
- scikit-learn (1.7.2)
- pandas (2.3.3)
- numpy (2.3.3)
- matplotlib (3.10.6)

## Usage

### Run All Algorithms
```bash
python main.py
```

This will:
- Load both datasets from the `data/` folder
- Run all six classification algorithms
- Perform 10-fold cross validation
- Display performance metrics (Accuracy, Precision, Recall, F1-Score)

## Dataset Format
- Format: Each line represents one data sample
- Last column: class label (0 or 1)
- Other columns: feature values (can be numeric or string)
- Automatic preprocessing handles mixed data types

## Key Features

### Reproducibility
- Fixed random seeds (random_state=42) throughout
- Deterministic algorithm behavior
- Consistent data preprocessing
- Same results across multiple runs

### Performance Metrics
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions
- **Recall**: Found positive instances
- **F1-Score**: Balance of precision and recall

### Modular Design
- Separate implementation for each algorithm
- Easy to extend with new algorithms

## Individual Contributions

### Alexander Lott (asl139)
- Implemented KNN, Decision Tree, and Naive Bayes algorithms
- Developed corresponding helper modules

### Alif Al Hasan (axh1218)
- Implemented SVM, AdaBoost, and Neural Network algorithms
- Developed data loader and evaluator helpers

## File Descriptions

### Root Level
- `main.py` - Main script that runs all algorithms
- `requirements.txt` - Python package dependencies
- `README.md` - Project overview file

### Individual Folders
- `axh1218/` - Alif's algorithms and helpers
- `asl139/` - Alex's algorithms and helpers

## Testing
Each algorithm can be tested individually by importing from the respective folders:

```python
# Test SVM
from axh1218.algorithms.svm import SVM
from axh1218.helpers.data_loader import DataLoader

data_loader = DataLoader()
X, y = data_loader.load_data('data/project1_dataset1.txt')

model = SVM()
model.fit(X, y)
predictions = model.predict(X)
```

## Support
For questions or issues, please contact the respective team members:
- Alif Al Hasan: axh1218@case.edu
- Alexander Lott: asl139@case.edu