"""
Direct Autoencoder Implementation using TensorFlow/Keras
This provides more control than PyOD's wrapper and supports custom architectures
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')


class DirectAutoencoder:
    """
    Direct autoencoder implementation for anomaly detection
    
    Architecture:
        Encoder: Input -> Dense layers -> Bottleneck (compressed)
        Decoder: Bottleneck -> Dense layers -> Output (reconstructed)
    """
    
    def __init__(self, input_dim, bottleneck_dim=None, hidden_dims=None, 
                 activation='relu', learning_rate=0.001, epochs=50, batch_size=32):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        bottleneck_dim : int
            Dimension of bottleneck (latent space). If None, uses input_dim // 4
        hidden_dims : list
            Dimensions of hidden layers. If None, uses [input_dim//2, bottleneck_dim]
        activation : str
            Activation function for hidden layers
        learning_rate : float
            Learning rate for optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim or max(2, input_dim // 4)
        self.hidden_dims = hidden_dims or [input_dim // 2]
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.encoder = None
        self.decoder = None
        self.reconstruction_errors = None
        self.threshold = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the autoencoder architecture"""
        
        # ===== ENCODER =====
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder: progressively compress
        x = input_layer
        encoder_layers = []
        
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            encoder_layers.append(dim)
        
        # Bottleneck (latent representation)
        bottleneck = layers.Dense(self.bottleneck_dim, activation=self.activation, name='bottleneck')(x)
        
        # ===== DECODER =====
        # Mirror the encoder structure
        x = bottleneck
        decoder_dims = self.hidden_dims[::-1]  # Reverse order
        
        for dim in decoder_dims:
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer (reconstruct original input)
        output_layer = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        # ===== FULL AUTOENCODER =====
        self.model = Model(input_layer, output_layer)
        
        # ===== ENCODER (for latent space) =====
        self.encoder = Model(input_layer, bottleneck)
        
        # ===== DECODER (for reconstruction) =====
        latent_input = layers.Input(shape=(self.bottleneck_dim,))
        x = latent_input
        for dim in decoder_dims:
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(x)
        self.decoder = Model(latent_input, decoded)
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    def fit(self, X, validation_split=0.1, verbose=1):
        """
        Train the autoencoder on normal data
        
        Parameters:
        -----------
        X : numpy array
            Training data (assumed to contain mostly normal samples)
        validation_split : float
            Fraction of data to use for validation
        verbose : int
            Verbosity level (0=quiet, 1=progress bar, 2=one line per epoch)
        
        Returns:
        --------
        history : keras History object
            Training history
        """
        print(f"Training autoencoder with {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Bottleneck dimension: {self.bottleneck_dim}")
        print(f"  Hidden layers: {self.hidden_dims}")
        
        history = self.model.fit(
            X, X,  # Input = Target (reconstruction)
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=True
        )
        
        # Compute reconstruction errors on training data
        self.reconstruction_errors = np.mean(np.square(X - self.model.predict(X, verbose=0)), axis=1)
        
        # Set threshold (e.g., 95th percentile of training errors)
        self.threshold = np.percentile(self.reconstruction_errors, 95)
        print(f"  Reconstruction error threshold (95th percentile): {self.threshold:.6f}")
        
        return history
    
    def predict(self, X):
        """
        Predict reconstruction errors and anomaly labels
        
        Parameters:
        -----------
        X : numpy array
            Data to predict on
        
        Returns:
        --------
        reconstruction_errors : numpy array
            Reconstruction error for each sample
        anomaly_labels : numpy array
            Binary labels (1 = anomaly, 0 = normal)
        """
        X_reconstructed = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - X_reconstructed), axis=1)
        anomaly_labels = (reconstruction_errors > self.threshold).astype(int)
        
        return reconstruction_errors, anomaly_labels
    
    def get_latent_representation(self, X):
        """
        Get the latent (bottleneck) representation of the data
        
        Parameters:
        -----------
        X : numpy array
            Input data
        
        Returns:
        --------
        latent_vectors : numpy array
            Compressed representation (shape: n_samples x bottleneck_dim)
        """
        return self.encoder.predict(X, verbose=0)
    
    def reconstruct(self, X):
        """
        Reconstruct input from latent space
        
        Parameters:
        -----------
        X : numpy array
            Input data
        
        Returns:
        --------
        X_reconstructed : numpy array
            Reconstructed data
        """
        return self.model.predict(X, verbose=0)
    
    def summary(self):
        """Print model architecture summary"""
        self.model.summary()


# ===== EXAMPLE USAGE =====

def example_anomaly_detection():
    """Example: Train and use autoencoder for anomaly detection"""
    
    # Generate synthetic data (normal + anomalies)
    np.random.seed(42)
    n_normal = 800
    n_anomaly = 200
    
    # Normal data: clustered around (0, 0)
    X_normal = np.random.normal(0, 1, (n_normal, 10))
    
    # Anomalies: random noise in extreme regions
    X_anomaly = np.random.uniform(-5, 5, (n_anomaly, 10))
    
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print("=" * 60)
    print("DIRECT AUTOENCODER FOR ANOMALY DETECTION")
    print("=" * 60)
    print(f"Data shape: {X.shape}")
    print(f"True anomalies: {int(np.sum(y_true))} / {len(y_true)}")
    
    # Create and train autoencoder
    ae = DirectAutoencoder(
        input_dim=10,
        bottleneck_dim=3,          # Compress to 3 dimensions
        hidden_dims=[8, 5],        # 10 -> 8 -> 5 -> 3 (bottleneck)
        epochs=50,
        batch_size=32
    )
    
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    ae.summary()
    
    # Train on all data (in practice, train on clean data only)
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = ae.fit(X, validation_split=0.1, verbose=1)
    
    # Predict
    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    reconstruction_errors, predictions = ae.predict(X)
    
    print(f"Detected anomalies: {int(np.sum(predictions))} / {len(predictions)}")
    print(f"Reconstruction error - Min: {np.min(reconstruction_errors):.6f}, "
          f"Max: {np.max(reconstruction_errors):.6f}, "
          f"Mean: {np.mean(reconstruction_errors):.6f}")
    
    # Evaluation (if ground truth available)
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    auc = roc_auc_score(y_true, reconstruction_errors)
    cm = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Confusion Matrix:")
    print(f"  True Negatives: {tn}, False Positives: {fp}")
    print(f"  False Negatives: {fn}, True Positives: {tp}")
    print(f"Precision: {tp / (tp + fp):.4f}, Recall: {tp / (tp + fn):.4f}")
    
    # Latent representation
    print("\n" + "=" * 60)
    print("LATENT SPACE")
    print("=" * 60)
    latent = ae.get_latent_representation(X)
    print(f"Latent representation shape: {latent.shape}")
    print(f"Latent space (first 5 samples):\n{latent[:5]}")


if __name__ == "__main__":
    example_anomaly_detection()
