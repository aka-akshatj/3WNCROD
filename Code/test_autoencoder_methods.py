"""
Test script: Run AutoEncoder comparison on your project datasets
Usage: python test_autoencoder_methods.py
"""

import os
import sys
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from autoencoder_direct import DirectAutoencoder


def load_dataset(dataset_path):
    """Load dataset from .mat or .csv"""
    file_ext = os.path.splitext(dataset_path)[1].lower()
    
    if file_ext == '.mat':
        load_data = loadmat(dataset_path)
        
        # Find the feature matrix (ignore dunder methods and common label names)
        ignore_keys = ['y', 'label', 'labels', 'outlier', 'outliers', 'ground_truth']
        data_keys = [k for k in load_data.keys() if not k.startswith('__') and k.lower() not in ignore_keys]
        X = load_data[data_keys[0]]
        
        # Smart search for the ground truth labels
        y = None
        for key in load_data.keys():
            if key.lower() in ignore_keys:
                y = load_data[key].flatten()
                break
                
        # If we STILL can't find labels, check if the last column of X is binary
        if y is None:
            unique_last_col = np.unique(X[:, -1])
            if len(unique_last_col) <= 2 and np.all(np.isin(unique_last_col, [0, 1])):
                print("  [Note] Extracting labels from the last column of the matrix.")
                y = X[:, -1]
                X = X[:, :-1]
            else:
                print(f"  [Warning] No labels found in {os.path.basename(dataset_path)}. AUC will be NaN.")
                y = np.zeros(X.shape[0])
                
    elif file_ext == '.csv':
        df = pd.read_csv(dataset_path)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
        X = df.values
        y = X[:, -1] 
        X = X[:, :-1]
    else:
        raise ValueError(f"Unsupported format: {file_ext}")
    
    # Preprocess
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def test_direct_autoencoder():
    """Test DirectAutoencoder on available datasets"""
    
    print("=" * 80)
    print("DIRECT AUTOENCODER TEST")
    print("=" * 80)
    
    # Find available datasets
    dataset_dirs = [
        'Datasets',
        'Code/Datasets'
    ]
    
    datasets_found = []
    for dataset_dir in dataset_dirs:
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                if file.endswith(('.mat', '.csv')):
                    datasets_found.append(os.path.join(dataset_dir, file))
    
    if not datasets_found:
        print("No datasets found in Datasets/ or Code/Datasets/")
        return
    
    # Test on first 2 datasets
    for i, dataset_path in enumerate(datasets_found[:2]):
        dataset_name = os.path.basename(dataset_path)
        
        print(f"\n{'='*80}")
        print(f"Test {i+1}: {dataset_name}")
        print('='*80)
        
        try:
            # Load
            X,y = load_dataset(dataset_path)
            print(f"Shape: {X.shape}")
            
            # Create autoencoder with adaptive parameters
            n_features = X.shape[1]
            bottleneck_dim = max(2, n_features // 4)
            hidden_dims = [max(4, n_features // 2)]
            
            print(f"Input features: {n_features}")
            print(f"Bottleneck dimension: {bottleneck_dim}")
            print(f"Hidden layers: {hidden_dims}")
            
            # Create and train
            ae = DirectAutoencoder(
                input_dim=n_features,
                bottleneck_dim=bottleneck_dim,
                hidden_dims=hidden_dims,
                epochs=50,  # Fewer epochs for testing
                batch_size=min(32, X.shape[0] // 5),
                learning_rate=0.001
            )
            
            print("\nTraining...")
            ae.fit(X, validation_split=0.1, verbose=0)
            
            # Predict
            print("Computing anomaly scores...")
            scores, labels = ae.predict(X)
            
            # Results
            n_anomalies = int(np.sum(labels))
            anomaly_rate = float(np.mean(labels))
            
            print(f"\nResults:")
            print(f"  Detected anomalies: {n_anomalies} ({anomaly_rate*100:.2f}%)")
            print(f"  Reconstruction error range: [{np.min(scores):.6f}, {np.max(scores):.6f}]")
            print(f"  Mean error: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
            print(f"  Anomaly threshold: {ae.threshold:.6f}")
            
            # Show top anomalies
            top_indices = np.argsort(scores)[-5:]
            print(f"\n  Top 5 anomalies (highest reconstruction errors):")
            for idx in top_indices[::-1]:
                print(f"    Sample {idx}: error = {scores[idx]:.6f}")
            
            # Get latent space
            latent = ae.get_latent_representation(X)
            print(f"\n  Latent space shape: {latent.shape}")
            print(f"  Latent representation is {n_features}D -> {latent.shape[1]}D compression")
            
            print(f"\n✓ {dataset_name} processed successfully")
            
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()


def compare_pyod_vs_direct(target_filename):
    """Compare PyOD vs Direct AutoEncoder on a specific dataset"""
    
    print("\n" + "=" * 80)
    print(f"COMPARISON: PyOD vs Direct AutoEncoder")
    print("=" * 80)
    
    # Search for the specific file
    dataset_dirs = ['Datasets', 'Code/Datasets']
    dataset_path = None
    
    for dataset_dir in dataset_dirs:
        potential_path = os.path.join(dataset_dir, target_filename)
        if os.path.exists(potential_path):
            dataset_path = potential_path
            break
            
    if not dataset_path:
        print(f"✗ Could not find {target_filename} in the Datasets folders.")
        return
    
    X, y = load_dataset(dataset_path)
    
    print(f"\nDataset: {target_filename}")
    print(f"Shape: {X.shape}\n")
    
    # ===== PyOD AutoEncoder =====
    print("1. PyOD AutoEncoder")
    print("-" * 80)
    try:
        from pyod.models.auto_encoder import AutoEncoder
        import time
        
        ae_params = {
            'hidden_neuron_list': [max(4, X.shape[1]//2), max(2, X.shape[1]//4)],
            'epoch_num': 30,
            'batch_size': 32,
            'verbose': 0,
            'preprocessing': True,
        }
        
        model_pyod = AutoEncoder(**ae_params)
        start = time.time()
        model_pyod.fit(X)
        pyod_time = time.time() - start
        
        pyod_scores = model_pyod.decision_scores_
        pyod_labels = model_pyod.labels_

        pyod_auc = roc_auc_score(y, pyod_scores)
        print(f"   PyOD ROC-AUC: {pyod_auc:.4f}")
        
        print(f"   Training time: {pyod_time:.3f}s")
        print(f"   Anomalies: {int(np.sum(pyod_labels))} ({np.mean(pyod_labels)*100:.2f}%)")
        print(f"   Score range: [{np.min(pyod_scores):.6f}, {np.max(pyod_scores):.6f}]")
        print(f"   ✓ Success")
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        pyod_scores = None
    
    # ===== Direct AutoEncoder =====
    print("\n2. Direct AutoEncoder (TensorFlow/Keras)")
    print("-" * 80)
    try:
        import time
        
        ae = DirectAutoencoder(
            input_dim=X.shape[1],
            bottleneck_dim=max(2, X.shape[1]//4),
            hidden_dims=[max(4, X.shape[1]//2)],
            epochs=30,  # Note: You can increase this to 50 for better results on big data
            batch_size=32,
            learning_rate=0.001
        )
        
        start = time.time()
        ae.fit(X, validation_split=0.1, verbose=0)
        direct_time = time.time() - start
        
        direct_scores, direct_labels = ae.predict(X)

        direct_auc = roc_auc_score(y, direct_scores)
        print(f"   Direct AE ROC-AUC: {direct_auc:.4f}")
        
        print(f"   Training time: {direct_time:.3f}s")
        print(f"   Anomalies: {int(np.sum(direct_labels))} ({np.mean(direct_labels)*100:.2f}%)")
        print(f"   Error range: [{np.min(direct_scores):.6f}, {np.max(direct_scores):.6f}]")
        print(f"   ✓ Success")
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        direct_scores = None
    
    # ===== Comparison =====
    print("\n3. Comparison Summary")
    print("-" * 80)
    
    if pyod_scores is not None and direct_scores is not None:
        # Normalize scores for comparison
        pyod_norm = (pyod_scores - np.min(pyod_scores)) / (np.max(pyod_scores) - np.min(pyod_scores))
        direct_norm = (direct_scores - np.min(direct_scores)) / (np.max(direct_scores) - np.min(direct_scores))
        
        correlation = np.corrcoef(pyod_norm, direct_norm)[0, 1]
        
        print(f"   Score correlation (normalized): {correlation:.4f}")
        print(f"   PyOD anomalies:     {int(np.sum(pyod_labels))}")
        print(f"   Direct anomalies:   {int(np.sum(direct_labels))}")
        print(f"   Difference:         {abs(int(np.sum(pyod_labels)) - int(np.sum(direct_labels)))}")
        
        if correlation > 0.7:
            print("\n   ✓ Both methods agree closely on anomaly detection")
        else:
            print("\n   ⚠ Methods show different anomaly patterns")
    else:
        print("   Could not complete comparison")


if __name__ == "__main__":
    print("\nAutoencoder Testing Suite - Large Datasets")
    print("=" * 80)
    
    # Add or remove datasets from this list to test them sequentially
    target_datasets = [
        'mammography.mat',    # The ultimate class imbalance test (11k rows)
        'annthyroid.mat',     # Solid medium-large dataset (7k rows)
        'musk.mat'            # High dimensionality test (166 features)
    ]
    
    for ds in target_datasets:
        compare_pyod_vs_direct(ds)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)