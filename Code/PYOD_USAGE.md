# PyOD Outlier Detection - Usage Guide

## Installation

First, install the required dependencies:

```bash
pip install -r ../requirements_pyod.txt
```

Or install PyOD directly:

```bash
pip install pyod numpy scipy scikit-learn
```

## Files

1. **`pyod_detection.py`**: Core module with PyOD detection functions
   - `load_and_preprocess_dataset()`: Loads and normalizes datasets
   - `detect_outliers_pyod()`: Detects outliers using a single PyOD algorithm
   - `process_dataset_with_multiple_algorithms()`: Processes a dataset with multiple algorithms
   - `get_top_anomalies()`: Gets top k anomalies

2. **`run_pyod.py`**: Runner script that processes all datasets
   - Processes all 10 datasets in the Datasets folder
   - Uses multiple PyOD algorithms (IForest, LOF, COPOD, HBOS, KNN)
   - Provides detailed timing and results

## Usage

### Run all datasets with all algorithms:

```bash
cd Code
python run_pyod.py
```

### Modify algorithms:

Edit `run_pyod.py` and change the `algorithms` list:

```python
algorithms = [
    'IForest',    # Isolation Forest
    'LOF',        # Local Outlier Factor
    'COPOD',      # Copula-based Outlier Detection
    'HBOS',       # Histogram-based Outlier Score
    'KNN'         # k-Nearest Neighbors
]
```

### Use a single algorithm on a single dataset:

```python
from pyod_detection import load_and_preprocess_dataset, detect_outliers_pyod

# Load dataset
data = load_and_preprocess_dataset('./Datasets/annthyroid.mat', 'annthyroid')

# Detect outliers
scores, labels, model, fit_time = detect_outliers_pyod(data, algorithm='IForest')

print(f"Detected {sum(labels)} outliers")
print(f"Fit time: {fit_time:.2f} seconds")
```

## Available Algorithms

- **IForest**: Isolation Forest (fast, good for high-dimensional data)
- **LOF**: Local Outlier Factor (good for local anomalies)
- **COPOD**: Copula-based Outlier Detection (fast, parameter-free)
- **HBOS**: Histogram-based Outlier Score (fast, good for large datasets)
- **KNN**: k-Nearest Neighbors (classical approach)
- **OCSVM**: One-Class SVM (good for non-linear boundaries)
- **ABOD**: Angle-Based Outlier Detection
- **PCA**: Principal Component Analysis based
- **AutoEncoder**: Deep learning based (slower, requires more parameters)

## Output

The runner script provides:
- Per-dataset results with timing
- Outlier statistics (count, percentage, score ranges)
- Top 10 anomalies for each algorithm
- Summary of total runtime and per-algorithm performance

## Performance

PyOD algorithms are generally much faster than WNCROD:
- IForest, COPOD, HBOS: Usually seconds to minutes per dataset
- LOF, KNN: Moderate speed (depends on dataset size)
- AutoEncoder: Slower (requires training neural network)

## Comparison with WNCROD

- **WNCROD**: Custom algorithm, slower but handles mixed data types
- **PyOD**: Standard algorithms, faster, primarily for numerical data
- Both can be used to compare results and performance

