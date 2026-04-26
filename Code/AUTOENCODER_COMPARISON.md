"""
Comparison: PyOD AutoEncoder vs Direct Implementation
And integration with existing WNCROD project
"""

# ============================================================================
# KEY DIFFERENCES
# ============================================================================

"""
1. PyOD AutoEncoder (Current)
   ---------
   • Pros:
     - Unified interface with other algorithms
     - No need to install TensorFlow separately (but AutoEncoder does require it)
     - Automatic parameter optimization for anomaly detection
     - Directly integrated into the existing pipeline
   
   • Cons:
     - Less control over architecture
     - Limited customization of layers/activation functions
     - Parameters are more rigid and predetermined
     - Black-box training process
   
   • Usage:
     from pyod.models.auto_encoder import AutoEncoder
     model = AutoEncoder(hidden_neuron_list=[64, 32], epoch_num=50)
     model.fit(X)
     outlier_scores = model.decision_scores_

2. Direct Implementation (TensorFlow/Keras)
   ---------
   • Pros:
     - Full control over architecture (number of layers, neurons, activations)
     - Can implement custom loss functions
     - Supports advanced techniques (attention, skip connections, etc.)
     - Can save/load specific encoder or decoder
     - Better for research/experimentation
     - Visualize latent space easily
   
   • Cons:
     - More code to write
     - Need to manage hyperparameters manually
     - Requires TensorFlow installed
     - Need to handle training details (validation split, early stopping, etc.)
   
   • Usage:
     from autoencoder_direct import DirectAutoencoder
     ae = DirectAutoencoder(input_dim=10, bottleneck_dim=3, hidden_dims=[8, 5])
     ae.fit(X)
     errors, labels = ae.predict(X)
"""


# ============================================================================
# INTEGRATION EXAMPLE: Using Direct Autoencoder with Your Dataset
# ============================================================================

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import os
from autoencoder_direct import DirectAutoencoder


def use_direct_autoencoder_on_dataset(dataset_path, dataset_name):
    """
    Example: Use DirectAutoencoder instead of PyOD's wrapper
    
    Parameters:
    -----------
    dataset_path : str
        Path to .mat or .csv file
    dataset_name : str
        Name of the dataset
    
    Returns:
    --------
    results : dict
        Anomaly detection results
    """
    
    # ===== LOAD AND PREPROCESS =====
    print(f"\nLoading dataset: {dataset_name}")
    
    file_ext = os.path.splitext(dataset_path)[1].lower()
    
    if file_ext == '.mat':
        load_data = loadmat(dataset_path)
        data_key = [k for k in load_data.keys() if not k.startswith('__')][0]
        X = load_data[data_key]
    elif file_ext == '.csv':
        df = pd.read_csv(dataset_path)
        # Convert categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
        X = df.values
    else:
        raise ValueError(f"Unsupported format: {file_ext}")
    
    # Handle NaN/Inf and normalize
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print(f"Data shape: {X.shape}")
    
    # ===== CREATE AND TRAIN AUTOENCODER =====
    n_features = X.shape[1]
    
    # Adaptive hyperparameters based on data size
    bottleneck_dim = max(2, n_features // 4)  # Compress to 1/4 of input
    hidden_dims = [max(4, n_features // 2), bottleneck_dim]  # One hidden layer
    
    print(f"\nAutoencoder Configuration:")
    print(f"  Input dimension: {n_features}")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Bottleneck dimension: {bottleneck_dim}")
    
    ae = DirectAutoencoder(
        input_dim=n_features,
        bottleneck_dim=bottleneck_dim,
        hidden_dims=hidden_dims,
        epochs=50,
        batch_size=min(32, X.shape[0] // 10),
        learning_rate=0.001
    )
    
    print("\nTraining autoencoder...")
    ae.fit(X, validation_split=0.1, verbose=0)
    
    # ===== PREDICT ANOMALIES =====
    print("Computing anomaly scores...")
    reconstruction_errors, predictions = ae.predict(X)
    
    # ===== RESULTS =====
    n_anomalies = int(np.sum(predictions))
    anomaly_rate = float(np.mean(predictions))
    
    results = {
        'dataset_name': dataset_name,
        'data_shape': X.shape,
        'reconstruction_errors': reconstruction_errors,
        'predictions': predictions,
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate,
        'threshold': ae.threshold,
        'min_error': float(np.min(reconstruction_errors)),
        'max_error': float(np.max(reconstruction_errors)),
        'mean_error': float(np.mean(reconstruction_errors)),
        'std_error': float(np.std(reconstruction_errors)),
        'encoder': ae.encoder,  # For latent space analysis
        'decoder': ae.decoder,  # For reconstruction
    }
    
    print(f"Detected {n_anomalies} anomalies ({anomaly_rate*100:.2f}%)")
    print(f"Reconstruction errors - Min: {results['min_error']:.6f}, "
          f"Max: {results['max_error']:.6f}, Mean: {results['mean_error']:.6f}")
    
    return results


# ============================================================================
# ADVANCED FEATURE: Compare PyOD vs Direct AutoEncoder
# ============================================================================

def compare_pyod_vs_direct(dataset_path, dataset_name):
    """
    Compare PyOD AutoEncoder with Direct Implementation
    
    Usage:
    ------
    results = compare_pyod_vs_direct('path/to/dataset.mat', 'thyroid')
    """
    
    # Load data
    file_ext = os.path.splitext(dataset_path)[1].lower()
    if file_ext == '.mat':
        load_data = loadmat(dataset_path)
        data_key = [k for k in load_data.keys() if not k.startswith('__')][0]
        X = load_data[data_key]
    elif file_ext == '.csv':
        df = pd.read_csv(dataset_path)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
        X = df.values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print(f"\nComparing PyOD vs Direct AutoEncoder on {dataset_name}")
    print("=" * 70)
    
    # ===== PyOD AutoEncoder =====
    print("\n1. PyOD AutoEncoder")
    print("-" * 70)
    try:
        from pyod.models.auto_encoder import AutoEncoder
        import time
        
        ae_params = {
            'hidden_neuron_list': [max(4, X.shape[1]//2), max(2, X.shape[1]//4)],
            'epoch_num': 50,
            'batch_size': 32,
            'verbose': 0,
            'preprocessing': True,
            'lr': 0.001,
            'optimizer_name': 'adam',
            'hidden_activation_name': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.2
        }
        
        model_pyod = AutoEncoder(**ae_params)
        start = time.time()
        model_pyod.fit(X)
        pyod_time = time.time() - start
        
        pyod_scores = model_pyod.decision_scores_
        pyod_labels = model_pyod.labels_
        
        print(f"Training time: {pyod_time:.2f}s")
        print(f"Anomalies detected: {int(np.sum(pyod_labels))} ({np.mean(pyod_labels)*100:.2f}%)")
        print(f"Decision scores - Min: {np.min(pyod_scores):.6f}, Max: {np.max(pyod_scores):.6f}")
        
    except ImportError as e:
        print(f"ERROR: {e}")
        pyod_scores = None
    
    # ===== Direct AutoEncoder =====
    print("\n2. Direct Implementation (TensorFlow/Keras)")
    print("-" * 70)
    try:
        import time
        
        ae_direct = DirectAutoencoder(
            input_dim=X.shape[1],
            bottleneck_dim=max(2, X.shape[1]//4),
            hidden_dims=[max(4, X.shape[1]//2)],
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
        
        start = time.time()
        ae_direct.fit(X, validation_split=0.1, verbose=0)
        direct_time = time.time() - start
        
        direct_scores, direct_labels = ae_direct.predict(X)
        
        print(f"Training time: {direct_time:.2f}s")
        print(f"Anomalies detected: {int(np.sum(direct_labels))} ({np.mean(direct_labels)*100:.2f}%)")
        print(f"Reconstruction errors - Min: {np.min(direct_scores):.6f}, Max: {np.max(direct_scores):.6f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        direct_scores = None
    
    # ===== COMPARISON SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if pyod_scores is not None and direct_scores is not None:
        # Correlation of anomaly scores
        correlation = np.corrcoef(pyod_scores, direct_scores)[0, 1]
        print(f"Score correlation: {correlation:.4f} (higher = more similar)")
        print(f"PyOD detected:   {int(np.sum(pyod_labels))} anomalies")
        print(f"Direct detected: {int(np.sum(direct_labels))} anomalies")
    
    return {
        'pyod_scores': pyod_scores,
        'direct_scores': direct_scores,
        'data': X
    }


# ============================================================================
# USAGE IN YOUR PROJECT
# ============================================================================

"""
Option 1: Use PyOD AutoEncoder (Current - Simpler)
    • Already integrated in run_pyod.py
    • Just run: python run_pyod.py
    • AutoEncoder included automatically

Option 2: Use Direct AutoEncoder (More Control)
    • For a single dataset:
    
        from autoencoder_direct import DirectAutoencoder
        from scipy.io import loadmat
        
        data = loadmat('Datasets/thyroid.mat')
        X = list(data.values())[0]
        
        ae = DirectAutoencoder(input_dim=X.shape[1], epochs=50)
        ae.fit(X, verbose=1)
        scores, labels = ae.predict(X)
        
    • To integrate with run_pyod.py:
      Add this function to run_pyod.py and include 'DirectAutoencoder' in 
      the algorithms list

Option 3: Compare Both (Experimental)
    • Run: python -c "from autoencoder_direct import compare_pyod_vs_direct; compare_pyod_vs_direct('Code/Datasets/thyroid.mat', 'thyroid')"
    • Helps understand differences in performance
"""
