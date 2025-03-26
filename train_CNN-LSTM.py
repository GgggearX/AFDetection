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
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from models.CNN_LSTM_model import get_model
from datasets import ECGSequence, load_data
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from tqdm import tqdm

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
# 自定义 F1
# --------------------------------
def f1_score(y_true, y_pred):
    # 使用 0.1 作为阈值
    y_pred_binary = K.cast(K.greater(y_pred, 0.1), K.floatx())
    true_positives = K.sum(y_true * y_pred_binary)
    possible_positives = K.sum(y_true)
    predicted_positives = K.sum(y_pred_binary)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

def f1_score_at_05(y_true, y_pred):
    # 使用 0.05 作为阈值
    y_pred_binary = K.cast(K.greater(y_pred, 0.05), K.floatx())
    true_positives = K.sum(y_true * y_pred_binary)
    possible_positives = K.sum(y_true)
    predicted_positives = K.sum(y_pred_binary)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# --------------------------------
# 训练单个 Fold
# --------------------------------
def train_fold(x_data, y_data, train_idx, val_idx, fold, args, base_dir):
    """训练单个折"""
    # 创建模型和日志目录
    model_dir = os.path.join(base_dir, 'models', f'fold_{fold}')
    log_dir = os.path.join(base_dir, 'logs', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 准备训练和验证数据
    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_val = x_data[val_idx]
    y_val = y_data[val_idx]
    
    # 检查数据分布
    train_pos_ratio = np.mean(y_train)
    val_pos_ratio = np.mean(y_val)
    print(f"\nFold {fold} 数据分布:")
    print(f"训练集大小: {len(y_train)}")
    print(f"训练集正样本比例: {train_pos_ratio:.4f}")
    print(f"验证集大小: {len(y_val)}")
    print(f"验证集正样本比例: {val_pos_ratio:.4f}")
    
    # 对训练集进行SMOTE过采样，增加k_neighbors参数
    n_samples, n_timesteps, n_features = x_train.shape
    x_train_reshaped = x_train.reshape(n_samples, -1)
    
    # 调整SMOTE参数，使采样更平衡
    n_positives = np.sum(y_train.flatten() == 1)
    n_negatives = np.sum(y_train.flatten() == 0)
    sampling_strategy = min(0.5, n_negatives / n_positives)  # 限制正样本比例不超过50%
    
    smote = SMOTE(random_state=42, 
                  k_neighbors=min(5, n_positives),
                  sampling_strategy=sampling_strategy)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_reshaped, y_train.flatten())
    
    # 重塑回原始形状
    x_train_resampled = x_train_resampled.reshape(-1, n_timesteps, n_features)
    y_train_resampled = y_train_resampled.reshape(-1, 1)
    
    # 计算类别权重（使用更平衡的方式）
    class_counts = np.bincount(y_train_resampled.flatten())
    total = len(y_train_resampled)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    # 创建数据生成器，启用数据增强
    train_seq = ECGSequence(
        x_train_resampled, y_train_resampled,
        batch_size=args.batch_size,
        shuffle=True,
        use_augmentation=True  # 启用数据增强
    )
    
    val_seq = ECGSequence(
        x_val, y_val,
        batch_size=args.batch_size,
        shuffle=False,
        scalers=train_seq.scalers,
        use_augmentation=False  # 验证集不使用数据增强
    )
    
    # 保存标准化参数
    scaler_path = os.path.join(model_dir, 'scalers.npy')
    train_seq.save_scalers(scaler_path)
    
    # 创建自定义回调来显示训练进度
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_pbar = None
            self.step_pbar = None
            
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            n_batches = len(train_seq)
            self.epoch_pbar = tqdm(total=n_batches, 
                                 position=0, 
                                 leave=True,
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - loss: {postfix}')
            
        def on_batch_end(self, batch, logs=None):
            if logs is None:
                logs = {}
            train_precision = logs.get('precision', 0)
            train_recall = logs.get('recall', 0)
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            metrics_str = f"{logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc', 0):.4f} - f1: {train_f1:.4f}"
            self.epoch_pbar.set_postfix_str(metrics_str)
            self.epoch_pbar.update(1)
            
        def on_epoch_end(self, epoch, logs=None):
            if self.epoch_pbar:
                self.epoch_pbar.close()
                print()  # 换行

    # 编译模型
    n_leads = x_train.shape[2]  # 获取导联数
    model = get_model(
        max_seq_length=x_data.shape[1],
        n_classes=1,
        last_layer='sigmoid',
        n_leads=n_leads
    )
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate * 0.5),  # 降低初始学习率
        loss=focal_loss(gamma=2.0, alpha=0.35),  # 降低alpha值，减少对正样本的过度关注
        metrics=['accuracy', AUC(), Precision(), Recall(), f1_score, f1_score_at_05]
    )
    
    # 设置回调函数
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.5,  # 更温和的学习率衰减
            patience=10,  # 增加patience
            mode='max',
            min_lr=1e-6
        ),
        ModelCheckpoint(
            os.path.join(model_dir, f'model_best_cnn_lstm_fold_{fold}.hdf5'),
            monitor='val_f1_score',
            save_best_only=True,
            mode='max'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        CSVLogger(
            os.path.join(log_dir, f'training_fold_{fold}.csv')
        ),
        ProgressCallback()
    ]
    
    # 训练模型
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
# 主函数
# --------------------------------
def main():
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 设置默认路径
    default_data_dir = 'data/physionet/training2017'
    default_reference_file = 'data/REFERENCE-v3.csv'
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练CNN-LSTM模型')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='数据目录路径')
    parser.add_argument('--reference_file', type=str, default=default_reference_file, help='标签文件路径')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')  # 增加训练轮数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')  # 降低初始学习率
    parser.add_argument('--max_seq_length', type=int, default=4096, help='最大序列长度')
    args = parser.parse_args()

    # 创建输出目录结构
    base_dir = 'CNN-LSTM'
    os.makedirs(base_dir, exist_ok=True)

    print("开始加载数据...")
    # 加载数据
    x_data, y_data = load_data(args.data_dir, args.reference_file, args.max_seq_length)
    
    # 显示整体数据集的正样本比例
    total_pos_ratio = np.mean(y_data)
    total_samples = len(y_data)
    total_pos_samples = np.sum(y_data)
    print(f"\n整体数据集信息:")
    print(f"总样本数: {total_samples}")
    print(f"正样本数: {total_pos_samples}")
    print(f"正样本比例: {total_pos_ratio:.4f}")

    # 创建分层K-fold交叉验证
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # 存储每个fold的结果
    fold_histories = []
    
    print(f"\n开始{args.n_splits}折交叉验证训练...")
    # 训练每个fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_data, y_data.flatten())):
        print(f"\n训练 Fold {fold+1}/{args.n_splits}")
        model, history = train_fold(x_data, y_data, train_idx, val_idx, fold+1, args, base_dir)
        fold_histories.append(history.history)
        print(f"Fold {fold+1} 训练完成")

    # 保存交叉验证结果
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 计算并保存平均性能指标
    metrics = ['loss', 'accuracy', 'auc', 'f1_score', 'f1_score_at_05', 
              'val_loss', 'val_accuracy', 'val_auc', 'val_f1_score', 'val_f1_score_at_05']
    avg_metrics = {metric: [] for metric in metrics}
    
    for history in fold_histories:
        for metric in metrics:
            if metric in history:
                avg_metrics[metric].append(history[metric][-1])
    
    summary = {metric: np.mean(values) for metric, values in avg_metrics.items() if values}
    
    with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n交叉验证平均性能:")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
        
    # 特别关注F1分数
    if 'f1_score' in summary:
        print(f"\n使用0.1阈值的F1分数: {summary['f1_score']:.4f}")
    if 'f1_score_at_05' in summary:
        print(f"使用0.05阈值的F1分数: {summary['f1_score_at_05']:.4f}")

if __name__ == "__main__":
    main()
