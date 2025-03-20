from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                     CSVLogger, EarlyStopping)
from sklearn.model_selection import KFold
from models.rnn_model import get_model
import argparse
from datasets import ECGSequence
import pandas as pd
import numpy as np
import os
import h5py
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
import tensorflow as tf
import json

def focal_loss(gamma=1.5, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def process_labels(csv_path, output_dir):
    """只保留房颤（AF）的标签"""
    df = pd.read_csv(csv_path)
    af_labels = df.iloc[:, 4:5]  # 只保留AF列
    output_path = os.path.join(output_dir, 'preprocessed', 'af_labels.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return af_labels.values, output_path

def create_model_save_path(base_dir, fold):
    """创建模型保存路径"""
    model_dir = os.path.join(base_dir, 'models', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def train_fold(x_data, y_data, train_idx, val_idx, fold, args, base_dir):
    """训练单个fold"""
    # 创建保存目录
    model_dir = create_model_save_path(base_dir, fold)
    log_dir = os.path.join(base_dir, 'logs', f'fold_{fold}')
    os.makedirs(log_dir, exist_ok=True)

    # 准备数据
    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]

    # 创建数据生成器
    train_seq = ECGSequence(x_train, y_train, batch_size=args.batch_size, is_training=True)
    val_seq = ECGSequence(x_val, y_val, batch_size=args.batch_size, is_training=False, scalers=train_seq.scalers)

    # 保存标准化参数
    scaler_path = os.path.join(model_dir, 'scalers.npy')
    print(f"Saving scalers to: {scaler_path}")
    train_seq.save_scalers(scaler_path)

    # 计算类别权重
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
    class_weight_dict = dict(zip(unique_classes, class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # 创建模型
    model = get_model()
    
    # 编译模型
    model.compile(
        optimizer=Adam(args.learning_rate),
        loss=focal_loss(gamma=1.5, alpha=0.25),
        metrics=['accuracy', 'AUC']
    )

    # 回调函数
    callbacks = [
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        # 学习率调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=8,
            min_lr=args.learning_rate/1000,
            min_delta=0.0001
        ),
        # 模型检查点
        ModelCheckpoint(
            os.path.join(model_dir, f'model_best_rnn_fold_{fold}.hdf5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # TensorBoard
        TensorBoard(log_dir=log_dir),
        # 训练日志
        CSVLogger(os.path.join(log_dir, 'training.log'))
    ]

    # 训练模型
    history = model.fit(
        train_seq,
        epochs=args.epochs,
        validation_data=val_seq,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )

    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train RNN model for AF detection with k-fold CV')
    parser.add_argument('path_to_hdf5', type=str,
                       help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                       help='path to csv file containing annotations')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='number of folds for cross validation')
    parser.add_argument('--epochs', type=int, default=250,
                       help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='initial learning rate')
    args = parser.parse_args()

    # 创建输出目录结构
    base_dir = 'RNN'
    os.makedirs(base_dir, exist_ok=True)

    print("开始加载数据...")
    # 加载数据
    with h5py.File(args.path_to_hdf5, 'r') as f:
        x_data = f['tracings'][:]
    
    # 处理标签
    y_data, _ = process_labels(args.path_to_csv, base_dir)

    # 创建K-fold交叉验证
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # 存储每个fold的结果
    fold_histories = []
    
    print(f"\n开始{args.n_splits}折交叉验证训练...")
    # 训练每个fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        print(f"\n训练 Fold {fold+1}/{args.n_splits}")
        model, history = train_fold(x_data, y_data, train_idx, val_idx, fold+1, args, base_dir)
        fold_histories.append(history.history)

    # 保存交叉验证结果
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 计算并保存平均性能指标
    metrics = ['loss', 'accuracy', 'auc', 'val_loss', 'val_accuracy', 'val_auc']
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

if __name__ == "__main__":
    main()

# 执行命令：(powershell)
# $env:TF_FORCE_GPU_ALLOW_GROWTH='true'
# $env:TF_GPU_ALLOCATOR='cuda_malloc_async'
# python train_rnn.py data/ecg_tracings.hdf5 data/annotations/gold_standard.csv --n_splits 5 --epochs 70 --batch_size 32 --learning_rate 0.001