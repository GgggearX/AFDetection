import argparse
import os
import h5py
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                       ReduceLROnPlateau, CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from models.CNN_LSTM_model import get_model
from datasets import ECGSequence, load_data
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from tqdm import tqdm
import random

# --------------------------------
# Focal Loss
# --------------------------------
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        pt_1 = tf.where(tf.equal(y_true, 1.0), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0.0), y_pred, tf.zeros_like(y_pred))
        loss_1 = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1 + epsilon)
        loss_0 = -(1.0 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0 + epsilon)
        return tf.reduce_mean(loss_1 + loss_0)
    return focal_loss_fixed

# --------------------------------
# Custom F1 Score
# --------------------------------
def f1_score(y_true, y_pred):
    y_pred_binary = K.cast(K.greater(y_pred, 0.5), K.floatx())
    true_positives = K.sum(y_true * y_pred_binary)
    possible_positives = K.sum(y_true)
    predicted_positives = K.sum(y_pred_binary)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# --------------------------------
# Training Progress Callback
# --------------------------------
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_seq):
        super().__init__()
        self.epoch_pbar = None
        self.train_seq = train_seq
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}")
        n_batches = len(self.train_seq)
        self.epoch_pbar = tqdm(total=n_batches, 
                             position=0, 
                             leave=True,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - loss: {postfix}')
            
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        metrics_str = f"{logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc', 0):.4f} - f1: {logs.get('f1_score', 0):.4f}"
        self.epoch_pbar.set_postfix_str(metrics_str)
        self.epoch_pbar.update(1)
            
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_pbar:
            self.epoch_pbar.close()
            print()

# --------------------------------
# Train Single Fold
# --------------------------------
def train_fold(x_data, y_data, train_idx, val_idx, fold, args, base_dir):
    # Reset random seeds for each fold
    np.random.seed(42 + fold)
    tf.random.set_seed(42 + fold)
    random.seed(42 + fold)
    
    # Create directories
    model_dir = os.path.join(base_dir, 'models', f'fold_{fold}')
    log_dir = os.path.join(base_dir, 'logs', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Prepare data
    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_val = x_data[val_idx]
    y_val = y_data[val_idx]
    
    # Print data distribution
    print(f"\nFold {fold} Data Distribution:")
    print(f"Train size: {len(y_train)}")
    print(f"Train positive ratio: {np.mean(y_train):.4f}")
    print(f"Val size: {len(y_val)}")
    print(f"Val positive ratio: {np.mean(y_val):.4f}")
    
    # SMOTE oversampling
    n_samples, n_timesteps, n_features = x_train.shape
    x_train_reshaped = x_train.reshape(n_samples, -1)
    
    n_positives = np.sum(y_train.flatten() == 1)
    n_negatives = np.sum(y_train.flatten() == 0)
    sampling_strategy = min(0.5, n_negatives / n_positives)  # 增加正样本比例上限
    
    smote = SMOTE(random_state=42, 
                  k_neighbors=min(5, n_positives),  # 增加k_neighbors
                  sampling_strategy=sampling_strategy)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_reshaped, y_train.flatten())
    x_train_resampled = x_train_resampled.reshape(-1, n_timesteps, n_features)
    y_train_resampled = y_train_resampled.reshape(-1, 1)
    
    # Calculate class weights
    class_counts = np.bincount(y_train_resampled.flatten())
    total = len(y_train_resampled)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    # Create data generators
    train_seq = ECGSequence(
        x_train_resampled, y_train_resampled,
        batch_size=args.batch_size,
        shuffle=True,
        use_augmentation=True
    )
    
    val_seq = ECGSequence(
        x_val, y_val,
        batch_size=args.batch_size,
        shuffle=False,
        scalers=train_seq.scalers,
        use_augmentation=False
    )
    
    # Save scalers
    train_seq.save_scalers(os.path.join(model_dir, 'scalers.npy'))
    
    # Create and compile model
    n_leads = x_train.shape[2]
    model = get_model(
        max_seq_length=x_data.shape[1],
        n_classes=1,
        last_layer='sigmoid',
        n_leads=n_leads
    )
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.75),  # 使用focal loss，增加alpha值
        metrics=['accuracy', AUC(), Precision(), Recall(), f1_score]
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            os.path.join(model_dir, f'model_best_cnn_lstm_fold_{fold}.hdf5'),
            monitor='val_loss',
            save_best_only=True
        ),
        CSVLogger(
            os.path.join(log_dir, f'training_fold_{fold}.csv')
        ),
        ProgressCallback(train_seq)
    ]
    
    # Train model
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0
    )
    
    return model, history

# --------------------------------
# Main Function
# --------------------------------
def main():
    # Set random seed
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Set default path
    default_data_dir = 'data/physionet/training2017'
    default_reference_file = 'data/REFERENCE-v3.csv'
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CNN-LSTM Model')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Data directory path')
    parser.add_argument('--reference_file', type=str, default=default_reference_file, help='Label file path')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of K-fold cross-validation splits')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')  # Increase training epochs
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')  # Decrease initial learning rate
    parser.add_argument('--max_seq_length', type=int, default=4096, help='Maximum sequence length')
    args = parser.parse_args()

    # Create output directory structure
    base_dir = 'CNN-LSTM'
    os.makedirs(base_dir, exist_ok=True)

    print("Starting to load data...")
    # Load data
    x_data, y_data = load_data(args.data_dir, args.reference_file, args.max_seq_length)
    
    # Show overall dataset positive ratio
    total_pos_ratio = np.mean(y_data)
    total_samples = len(y_data)
    total_pos_samples = np.sum(y_data)
    print(f"\nOverall dataset information:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {total_pos_samples}")
    print(f"Positive ratio: {total_pos_ratio:.4f}")

    # Create K-fold cross-validation
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # Store results for each fold
    fold_histories = []
    
    print(f"\nStarting {args.n_splits} fold cross-validation training...")
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        print(f"\nTraining Fold {fold+1}/{args.n_splits}")
        
        # Check data distribution for current fold
        train_pos_ratio = np.mean(y_data[train_idx])
        val_pos_ratio = np.mean(y_data[val_idx])
        print(f"\nFold {fold+1} Data Distribution:")
        print(f"Train size: {len(train_idx)}")
        print(f"Train positive ratio: {train_pos_ratio:.4f}")
        print(f"Val size: {len(val_idx)}")
        print(f"Val positive ratio: {val_pos_ratio:.4f}")
        
        model, history = train_fold(x_data, y_data, train_idx, val_idx, fold+1, args, base_dir)
        fold_histories.append(history.history)
        print(f"Fold {fold+1} Training completed")

    # Save cross-validation results
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate and save average performance metrics
    metrics = ['loss', 'accuracy', 'auc', 'f1_score', 'val_loss', 'val_accuracy', 'val_auc', 'val_f1_score']
    avg_metrics = {metric: [] for metric in metrics}
    
    for history in fold_histories:
        for metric in metrics:
            if metric in history:
                avg_metrics[metric].append(history[metric][-1])
    
    summary = {metric: np.mean(values) for metric, values in avg_metrics.items() if values}
    
    with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\nAverage performance across cross-validation folds:")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
