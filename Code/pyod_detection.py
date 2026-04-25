"""
PyOD-based Outlier Detection
This module provides functions to detect outliers using various PyOD algorithms
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import time

# PyOD imports - common algorithms
PYOD_AVAILABLE = False
AUTOENCODER_AVAILABLE = False

try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.copod import COPOD
    from pyod.models.abod import ABOD
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    from pyod.models.pca import PCA
    PYOD_AVAILABLE = True
    
    try:
        from pyod.models.auto_encoder import AutoEncoder
        AUTOENCODER_AVAILABLE = True
    except ImportError:
        AUTOENCODER_AVAILABLE = False
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: PyOD not installed. Install with: pip install pyod")


def load_and_preprocess_dataset(dataset_path, dataset_name):
    """
    Load and preprocess a dataset from .mat file
    
    Parameters:
    -----------
    dataset_path : str
        Path to the .mat file
    dataset_name : str
        Name of the dataset
    
    Returns:
    --------
    data : numpy array
        Preprocessed data (normalized)
    """
    # Load dataset
    load_data = loadmat(dataset_path)
    
    # Get the data (variable name is usually the dataset name)
    data_key = [k for k in load_data.keys() if not k.startswith('__')][0]
    trandata = load_data[data_key]
    
    # Normalize numerical columns to [0,1]
    scaler = MinMaxScaler()
    trandata_normalized = scaler.fit_transform(trandata)
    
    return trandata_normalized


def detect_outliers_pyod(data, algorithm='IForest', **kwargs):
    """
    Detect outliers using PyOD algorithms
    
    Parameters:
    -----------
    data : numpy array
        Input data (rows: samples, columns: features)
    algorithm : str
        Algorithm name: 'IForest', 'LOF', 'OCSVM', 'COPOD', 'ABOD', 
                       'HBOS', 'KNN', 'PCA', 'AutoEncoder'
    **kwargs : dict
        Additional parameters for the algorithm
    
    Returns:
    --------
    outlier_scores : numpy array
        Outlier scores for each sample (higher = more anomalous)
    labels : numpy array
        Binary labels (1 = outlier, 0 = inlier)
    model : object
        Fitted model
    fit_time : float
        Time taken to fit the model
    """
    if not PYOD_AVAILABLE:
        raise ImportError("PyOD is not installed. Install with: pip install pyod")
    
    # Initialize algorithm
    algorithm = algorithm.upper()
    
    if algorithm == 'IFOREST':
        model = IForest(**kwargs)
    elif algorithm == 'LOF':
        model = LOF(**kwargs)
    elif algorithm == 'OCSVM':
        model = OCSVM(**kwargs)
    elif algorithm == 'COPOD':
        model = COPOD(**kwargs)
    elif algorithm == 'ABOD':
        model = ABOD(**kwargs)
    elif algorithm == 'HBOS':
        model = HBOS(**kwargs)
    elif algorithm == 'KNN':
        model = KNN(**kwargs)
    elif algorithm == 'PCA':
        model = PCA(**kwargs)
    elif algorithm == 'AUTOENCODER':
        if not AUTOENCODER_AVAILABLE:
            raise ImportError("AutoEncoder requires tensorflow/keras. Install with: pip install tensorflow")
        # AutoEncoder requires specific parameters
        ae_params = {
            'hidden_neurons': [data.shape[1]//2, data.shape[1]//4],
            'epochs': 50,
            'batch_size': 32,
            'verbose': 0
        }
        ae_params.update(kwargs)
        model = AutoEncoder(**ae_params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Fit and predict
    start_time = time.time()
    model.fit(data)
    fit_time = time.time() - start_time
    
    # Get outlier scores and labels
    outlier_scores = model.decision_scores_
    labels = model.labels_
    
    return outlier_scores, labels, model, fit_time


def process_dataset_with_multiple_algorithms(dataset_path, dataset_name, algorithms=None):
    """
    Process a dataset with multiple PyOD algorithms
    
    Parameters:
    -----------
    dataset_path : str
        Path to the .mat file
    dataset_name : str
        Name of the dataset
    algorithms : list
        List of algorithm names to use. If None, uses default set.
    
    Returns:
    --------
    results : dict
        Dictionary containing results for each algorithm
    """
    if algorithms is None:
        algorithms = ['IForest', 'LOF', 'COPOD', 'HBOS', 'KNN']
    
    # Load and preprocess
    print(f"Loading dataset: {dataset_name}")
    data = load_and_preprocess_dataset(dataset_path, dataset_name)
    print(f"Data shape: {data.shape}")
    
    results = {
        'dataset_name': dataset_name,
        'data_shape': data.shape,
        'algorithms': {}
    }
    
    # Process with each algorithm
    for algo in algorithms:
        try:
            print(f"\n  Running {algo}...")
            
            # Set default parameters based on data size
            algo_kwargs = {}
            if algo.upper() == 'LOF' or algo.upper() == 'KNN':
                # Use smaller n_neighbors for large datasets
                n_neighbors = min(20, data.shape[0] // 10)
                algo_kwargs['n_neighbors'] = max(5, n_neighbors)
            elif algo.upper() == 'IFOREST':
                algo_kwargs['n_estimators'] = 100
                algo_kwargs['max_samples'] = min(256, data.shape[0])
            
            outlier_scores, labels, model, fit_time = detect_outliers_pyod(
                data, algorithm=algo, **algo_kwargs
            )
            
            results['algorithms'][algo] = {
                'outlier_scores': outlier_scores,
                'labels': labels,
                'fit_time': fit_time,
                'n_outliers': int(np.sum(labels)),
                'outlier_rate': float(np.mean(labels)),
                'min_score': float(np.min(outlier_scores)),
                'max_score': float(np.max(outlier_scores)),
                'mean_score': float(np.mean(outlier_scores)),
                'std_score': float(np.std(outlier_scores))
            }
            
            print(f"    Completed in {fit_time:.2f} seconds")
            print(f"    Detected {int(np.sum(labels))} outliers ({np.mean(labels)*100:.2f}%)")
            
        except Exception as e:
            print(f"    ERROR with {algo}: {str(e)}")
            results['algorithms'][algo] = {'error': str(e)}
    
    return results


def get_top_anomalies(outlier_scores, top_k=10):
    """
    Get top k anomalies based on outlier scores
    
    Parameters:
    -----------
    outlier_scores : numpy array
        Outlier scores
    top_k : int
        Number of top anomalies to return
    
    Returns:
    --------
    top_indices : numpy array
        Indices of top anomalies
    top_scores : numpy array
        Scores of top anomalies
    """
    top_k = min(top_k, len(outlier_scores))
    top_indices = np.argsort(outlier_scores)[-top_k:][::-1]
    top_scores = outlier_scores[top_indices]
    return top_indices, top_scores


if __name__ == "__main__":
    # Example usage
    if PYOD_AVAILABLE:
        dataset_path = './Datasets/annthyroid.mat'
        dataset_name = 'annthyroid'
        
        results = process_dataset_with_multiple_algorithms(
            dataset_path, 
            dataset_name,
            algorithms=['IForest', 'LOF', 'COPOD']
        )
        
        print("\n" + "="*50)
        print("Results Summary")
        print("="*50)
        for algo, algo_results in results['algorithms'].items():
            if 'error' not in algo_results:
                print(f"\n{algo}:")
                print(f"  Fit time: {algo_results['fit_time']:.2f}s")
                print(f"  Outliers detected: {algo_results['n_outliers']}")
                print(f"  Score range: [{algo_results['min_score']:.4f}, {algo_results['max_score']:.4f}]")
    else:
        print("Please install PyOD: pip install pyod")

