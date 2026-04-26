"""
SUMMARY: Autoencoders in Your Project
================================================================================
"""

# ============================================================================
# CURRENT USAGE IN PROJECT
# ============================================================================

1. PyOD AutoEncoder (Currently Active)
   ---------
   Location: Code/pyod_detection.py (lines 126-152)
   
   How it works in your pipeline:
   
   • Automatically used when you run: python run_pyod.py
   • Part of the algorithm list: ['IForest', 'LOF', 'COPOD', 'HBOS', 'KNN', 'AutoEncoder']
   • Configuration (optimized for PyOD v2.0.6):
     - hidden_neuron_list: Progressively compresses input (e.g., [64, 32])
     - epoch_num: 50 training iterations
     - batch_size: 32 samples per update
     - batch_norm: True (stabilizes training)
     - dropout_rate: 0.2 (prevents overfitting)
     - optimizer: Adam with lr=0.001
   
   • Anomaly Detection:
     - Trains on normal data to learn reconstruction
     - High reconstruction error = likely anomaly
     - Output: decision_scores (errors) and labels (0/1)
   
   • Results:
     - Integrated into comparison reports
     - Generates output_logs/pyod_output_*.txt with all algorithms
     - Can be directly compared with 3WNCROD results


2. Direct Implementation (New Option)
   ---------
   Location: Code/autoencoder_direct.py (newly created)
   
   Key advantages:
   • Full control over architecture (number of layers, neurons, activations)
   • Can implement custom training strategies
   • Easy to visualize latent space (compressed representation)
   • Better for research and experimentation
   
   Basic usage:
   
   from autoencoder_direct import DirectAutoencoder
   
   # Create autoencoder
   ae = DirectAutoencoder(
       input_dim=22,              # Your dataset features
       bottleneck_dim=5,          # Compressed to 5 dimensions
       hidden_dims=[11, 8],       # 22 -> 11 -> 8 -> 5 (bottleneck)
       epochs=50,
       batch_size=32
   )
   
   # Train on normal data
   ae.fit(X_train, validation_split=0.1, verbose=1)
   
   # Detect anomalies
   reconstruction_errors, anomaly_labels = ae.predict(X_test)
   
   # Get latent representation
   latent_vectors = ae.get_latent_representation(X_test)


# ============================================================================
# ARCHITECTURE COMPARISON
# ============================================================================

PyOD AutoEncoder:
    Input (22D)
         ↓
    Dense 11 + BatchNorm + Dropout
         ↓
    Dense 8 + BatchNorm + Dropout
         ↓
    Bottleneck (5D) ← Compressed representation
         ↓
    Dense 8 + BatchNorm + Dropout
         ↓
    Dense 11 + BatchNorm + Dropout
         ↓
    Output (22D) ← Reconstructed input
    
    Loss = MSE(Input, Output)
    
    Anomaly Score = MSE(Input, Output)
    Label = 1 if Score > Threshold else 0


Direct AutoEncoder:
    [Identical architecture, but you control every layer]
    
    Can customize:
    - Number and size of hidden layers
    - Activation functions (relu, tanh, elu, etc.)
    - Regularization (dropout, batch_norm)
    - Loss functions (MSE, MAE, Huber, etc.)
    - Learning rate schedule
    - Early stopping criteria


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

EXAMPLE 1: Use PyOD AutoEncoder (Easiest)
-------------------------------------------

from pyod_detection import detect_outliers_pyod, load_and_preprocess_dataset

# Load your dataset
data = load_and_preprocess_dataset('Datasets/thyroid.mat', 'thyroid')

# Run AutoEncoder
outlier_scores, labels, model, fit_time = detect_outliers_pyod(
    data, 
    algorithm='AutoEncoder'
)

print(f"Detected {int(sum(labels))} anomalies in {fit_time:.2f}s")


EXAMPLE 2: Use Direct AutoEncoder (Most Control)
-------------------------------------------

from autoencoder_direct import DirectAutoencoder
import numpy as np

# Load your dataset
data = load_and_preprocess_dataset('Datasets/thyroid.mat', 'thyroid')

# Create autoencoder
ae = DirectAutoencoder(
    input_dim=data.shape[1],
    bottleneck_dim=max(2, data.shape[1] // 4),
    epochs=50,
    batch_size=32
)

# Train
ae.fit(data, validation_split=0.1, verbose=1)

# Detect anomalies
scores, labels = ae.predict(data)

# Analyze latent space
latent = ae.get_latent_representation(data)
print(f"Data compressed from {data.shape[1]}D to {latent.shape[1]}D")


EXAMPLE 3: Compare PyOD vs Direct on Same Data
-------------------------------------------

from autoencoder_direct import DirectAutoencoder
from pyod_detection import detect_outliers_pyod, load_and_preprocess_dataset

data = load_and_preprocess_dataset('Datasets/thyroid.mat', 'thyroid')

# PyOD version
pyod_scores, pyod_labels, _, _ = detect_outliers_pyod(data, 'AutoEncoder')

# Direct version
ae = DirectAutoencoder(input_dim=data.shape[1], epochs=50)
ae.fit(data, validation_split=0.1, verbose=0)
direct_scores, direct_labels = ae.predict(data)

# Compare
print(f"PyOD detected:    {sum(pyod_labels)} anomalies")
print(f"Direct detected:  {sum(direct_labels)} anomalies")

# Correlation of scores
correlation = np.corrcoef(pyod_scores, direct_scores)[0, 1]
print(f"Score correlation: {correlation:.3f}")


# ============================================================================
# WHEN TO USE WHICH
# ============================================================================

Use PyOD AutoEncoder when:
  ✓ You want consistency with other algorithms
  ✓ You need quick deployment in the existing pipeline
  ✓ You don't need to customize architecture
  ✓ You want unified reporting with all other methods

Use Direct AutoEncoder when:
  ✓ You want to experiment with different architectures
  ✓ You need to analyze the latent space
  ✓ You want custom loss functions or training strategies
  ✓ You're doing research or publishing results
  ✓ You need to save/load encoder/decoder separately


# ============================================================================
# KEY PARAMETERS TO TUNE
# ============================================================================

1. Bottleneck Dimension (Compression Level)
   - Smaller: More aggressive compression, faster training
   - Larger: Less compression, captures more detail
   - Recommendation: input_dim / 4 to input_dim / 2
   
   Example:
   - 7 features → bottleneck = 2
   - 22 features → bottleneck = 5-6
   - 100 features → bottleneck = 20-25

2. Hidden Layer Dimensions
   - Should progressively decrease toward bottleneck
   - Example progression: 22 → 16 → 10 → 6 → bottleneck
   
3. Epochs (Training Iterations)
   - Higher: Better learning, slower training
   - Lower: Faster training, may underfit
   - Start with 50, increase if validation loss improves

4. Batch Size
   - Larger: Faster training, less noise in gradients
   - Smaller: More frequent updates, noisier gradients
   - Recommendation: 32 (good balance)

5. Learning Rate
   - Higher: Faster convergence, may overshoot
   - Lower: Slower convergence, more stable
   - Recommendation: 0.001-0.01

6. Dropout Rate
   - Prevents overfitting
   - Typical: 0.1-0.3
   - Recommendation: 0.2 (works well for most datasets)

7. Anomaly Threshold
   - Percentile of reconstruction errors (typically 95%)
   - Lower: More samples flagged as anomalies
   - Higher: Fewer samples flagged as anomalies


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

Problem: Low anomaly detection rate
Solution:
  - Increase threshold percentile (90% instead of 95%)
  - Reduce bottleneck dimension (more compression)
  - Train longer (increase epochs)

Problem: Too many false positives
Solution:
  - Increase threshold percentile (95% → 98%)
  - Increase bottleneck dimension (less compression)
  - Add more training data

Problem: Very slow training
Solution:
  - Reduce number of epochs
  - Increase batch size (32 → 64)
  - Reduce hidden layer sizes

Problem: Unstable training (loss jumps around)
Solution:
  - Reduce learning rate (0.001 → 0.0001)
  - Increase batch size
  - Enable batch normalization


# ============================================================================
# INTERPRETING RESULTS
# ============================================================================

Reconstruction Error: How well the autoencoder reconstructs each sample

Low Error (≤ 0.01):    Normal samples (well-learned pattern)
Medium Error (0.01-0.05): Borderline samples (slightly unusual)
High Error (≥ 0.05):   Anomalous samples (very different from training data)

Latent Space (Bottleneck):
- Represents compressed features
- Dimensionality reduction benefit
- Can be visualized if bottleneck ≤ 3 dimensions
- Used for clustering or downstream analysis

Example interpretation:
  Sample 1: Error = 0.002 → Normal
  Sample 2: Error = 0.045 → Possible anomaly
  Sample 3: Error = 0.150 → Likely anomaly


# ============================================================================
# INTEGRATION WITH YOUR PROJECT
# ============================================================================

Current Pipeline:
    raw data → preprocess → PyOD AutoEncoder → detect anomalies → report

Your Options:

Option A (Current - No Changes):
    • Use PyOD AutoEncoder as-is
    • Already working and integrated
    • Run: python run_pyod.py

Option B (Add Direct AutoEncoder):
    • Modify run_pyod.py to include direct implementation
    • Compare results with PyOD
    • Choose better performer for your use case

Option C (Research):
    • Use Direct AutoEncoder for custom experiments
    • Save findings in separate analysis
    • Report results independently

RECOMMENDATION: Start with Option A (current PyOD setup), then try Option B 
if you want to explore or need custom architecture for specific use cases.
