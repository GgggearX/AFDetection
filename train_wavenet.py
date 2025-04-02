import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from datasets import ECGSequence, load_data
from models.wavenet_model import create_wavenet_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import seaborn as sns
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pandas as pd

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
        metrics_str = f"{logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc_1', 0):.4f}"
        self.epoch_pbar.set_postfix_str(metrics_str)
        self.epoch_pbar.update(1)
            
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_pbar:
            self.epoch_pbar.close()
            print()

def train_wavenet(data_dir, reference_file, output_dir, batch_size=32, epochs=100, 
                 max_seq_length=1024, validation_split=0.2):
    """
    训练WaveNet模型
    
    Args:
        data_dir: 数据目录
        reference_file: 标签文件路径
        output_dir: 输出目录
        batch_size: 批次大小
        epochs: 训练轮数
        max_seq_length: 最大序列长度
        validation_split: 验证集比例
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    x_data, y_data = load_data(data_dir, reference_file, max_seq_length)
    
    # 划分训练集和验证集
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - validation_split))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_val = x_data[val_indices]
    y_val = y_data[val_indices]
    
    # 打印数据分布
    print("\n数据分布:")
    print(f"训练集大小: {len(y_train)}")
    print(f"训练集正样本比例: {np.mean(y_train):.4f}")
    print(f"验证集大小: {len(y_val)}")
    print(f"验证集正样本比例: {np.mean(y_val):.4f}")
    
    # 创建数据生成器
    train_sequence = ECGSequence(x_train, y_train, batch_size=batch_size, 
                               use_augmentation=True)
    val_sequence = ECGSequence(x_val, y_val, batch_size=batch_size, 
                             use_augmentation=False)
    
    # 创建模型
    print("\n创建模型...")
    model = create_wavenet_model(input_shape=(max_seq_length, x_data.shape[2]))
    
    # 计算类别权重
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train.flatten()
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
    print("\n类别权重:")
    for cls, weight in class_weight_dict.items():
        print(f"类别 {cls}: {weight:.4f}")
    
    # 设置回调函数
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'checkpoints', f'wavenet_model_{timestamp}.h5'),
            monitor='val_auc_1',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_auc_1',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs', timestamp),
            histogram_freq=1
        ),
        CSVLogger(
            os.path.join(output_dir, 'logs', f'wavenet_history_{timestamp}.csv')
        ),
        ProgressCallback(train_sequence)
    ]
    
    # 训练模型
    print("\n开始训练...")
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0
    )
    
    # 保存训练历史
    np.save(os.path.join(output_dir, f'wavenet_history_{timestamp}.npy'), history.history)
    
    # 绘制训练曲线
    plot_training_history(history, output_dir, timestamp)
    
    # 评估模型
    evaluate_model(model, val_sequence, output_dir, timestamp)
    
    return model, history

def plot_training_history(history, output_dir, timestamp):
    """Plot training history curves"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot AUC curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc_1'], label='Training AUC')
    plt.plot(history.history['val_auc_1'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'wavenet_training_history_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, val_sequence, output_dir, timestamp):
    """Evaluate model performance"""
    # Get predictions
    y_pred = model.predict(val_sequence)
    y_true = np.concatenate([y for x, y in val_sequence], axis=0)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Calculate F1 score
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"AUC: {roc_auc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'wavenet_roc_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, f'wavenet_confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_wavenet_kfold(data_dir, reference_file, output_dir, n_splits=5, batch_size=32, epochs=100, 
                       max_seq_length=1024):
    """
    Train WaveNet model with K-fold cross validation
    
    Args:
        data_dir: Data directory
        reference_file: Label file path
        output_dir: Output directory
        n_splits: Number of folds
        batch_size: Batch size
        epochs: Number of epochs
        max_seq_length: Maximum sequence length
    """
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(output_dir, f'wavenet_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'plots'), exist_ok=True)
    
    # Load data
    print("Loading data...")
    x_data, y_data = load_data(data_dir, reference_file, max_seq_length)
    
    # Initialize K-fold cross validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    
    # Train model for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_data), 1):
        print(f"\nTraining Fold {fold}/{n_splits}")
        
        # Split data
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_val = x_data[val_idx]
        y_val = y_data[val_idx]
        
        # Print data distribution
        print("\nData Distribution:")
        print(f"Training set size: {len(y_train)}")
        print(f"Training set positive ratio: {np.mean(y_train):.4f}")
        print(f"Validation set size: {len(y_val)}")
        print(f"Validation set positive ratio: {np.mean(y_val):.4f}")
        
        # Create data generators
        train_sequence = ECGSequence(x_train, y_train, batch_size=batch_size, 
                                   use_augmentation=True)
        val_sequence = ECGSequence(x_val, y_val, batch_size=batch_size, 
                                 use_augmentation=False)
        
        # Create model
        print("\nCreating model...")
        model = create_wavenet_model(input_shape=(max_seq_length, x_data.shape[2]))
        
        # Calculate class weights
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train.flatten()
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        print("\nClass weights:")
        for cls, weight in class_weight_dict.items():
            print(f"Class {cls}: {weight:.4f}")
        
        # Set up callbacks
        fold_dir = os.path.join(model_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                os.path.join(fold_dir, 'best_model.h5'),
                monitor='val_auc_1',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_auc_1',
                patience=10,
                restore_best_weights=True,
                mode='max'
            ),
            TensorBoard(
                log_dir=os.path.join(fold_dir, 'logs'),
                histogram_freq=1
            ),
            CSVLogger(
                os.path.join(fold_dir, 'training_history.csv')
            ),
            ProgressCallback(train_sequence)
        ]
        
        # Train model
        print("\nStarting training...")
        history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # Save training history
        np.save(os.path.join(fold_dir, 'training_history.npy'), history.history)
        
        # Plot training curves
        plot_training_history(history, fold_dir, f'fold_{fold}')
        
        # Evaluate model
        y_pred = model.predict(val_sequence)
        y_true = np.concatenate([y for x, y in val_sequence], axis=0)
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        y_pred_binary = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        
        # Store results
        fold_results.append({
            'fold': fold,
            'auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"AUC: {roc_auc:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        
        # Plot ROC curve and confusion matrix
        evaluate_model(model, val_sequence, fold_dir, f'fold_{fold}')
    
    # Save overall results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(model_dir, 'fold_results.csv'), index=False)
    
    # Print average results
    print("\nAverage Results Across All Folds:")
    print(f"Average AUC: {np.mean([r['auc'] for r in fold_results]):.3f} ± {np.std([r['auc'] for r in fold_results]):.3f}")
    print(f"Average F1 Score: {np.mean([r['f1'] for r in fold_results]):.3f} ± {np.std([r['f1'] for r in fold_results]):.3f}")
    print(f"Average Precision: {np.mean([r['precision'] for r in fold_results]):.3f} ± {np.std([r['precision'] for r in fold_results]):.3f}")
    print(f"Average Recall: {np.mean([r['recall'] for r in fold_results]):.3f} ± {np.std([r['recall'] for r in fold_results]):.3f}")
    
    return fold_results

if __name__ == "__main__":
    # Set parameters
    DATA_DIR = "data/physionet/training2017"
    REFERENCE_FILE = "data/physionet/training2017/REFERENCE.csv"
    OUTPUT_DIR = "WaveNet"
    BATCH_SIZE = 32
    EPOCHS = 100
    MAX_SEQ_LENGTH = 1024
    N_SPLITS = 5
    
    # Train model with K-fold cross validation
    results = train_wavenet_kfold(
        data_dir=DATA_DIR,
        reference_file=REFERENCE_FILE,
        output_dir=OUTPUT_DIR,
        n_splits=N_SPLITS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_seq_length=MAX_SEQ_LENGTH
    ) 