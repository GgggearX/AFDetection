import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from datasets import ECGSequence, load_data
from models.wavenet_model import get_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
from tqdm import tqdm
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import json

def focal_loss(gamma=2.0, alpha=0.5):
    """实现 Focal Loss"""
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

class F1Score(tf.keras.metrics.Metric):
    """F1分数指标"""
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = Precision(thresholds=threshold)
        self.recall = Recall(thresholds=threshold)
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + K.epsilon() + r))
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

def create_model_save_path(base_dir, fold):
    model_dir = os.path.join(base_dir, 'models', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.epoch_pbar = None
        self.total_epochs = total_epochs
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")
        n_batches = self.params.get('steps', 0)
        self.epoch_pbar = tqdm(total=n_batches, position=0, leave=True,
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - loss: {postfix}')
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        train_prec = logs.get('precision_1', 0)
        train_rec = logs.get('recall_1', 0)
        train_f1 = 2 * (train_prec * train_rec) / (train_prec + train_rec) if (train_prec + train_rec) > 0 else 0
        metrics_str = f"loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc_1', 0):.4f} - f1: {train_f1:.4f}"
        self.epoch_pbar.set_postfix_str(metrics_str)
        self.epoch_pbar.update(1)
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_pbar:
            self.epoch_pbar.close()
            print()
        if logs:
            val_metrics_str = (f"val_acc: {logs.get('val_accuracy', 0):.4f} | "
                            f"val_auc: {logs.get('val_auc_1', 0):.4f} | "
                            f"val_f1: {logs.get('val_f1_score', 0):.4f} | "
                            f"val_prec: {logs.get('val_precision_1', 0):.4f} | "
                            f"val_recall: {logs.get('val_recall_1', 0):.4f}")
            print(f"[验证集] {val_metrics_str}")

def train_fold(x_train, y_train, x_val, y_val, fold, args):
    print(f"\n开始训练 fold {fold}")
    model_dir = create_model_save_path(args.model_dir, fold)
    log_dir = os.path.join(args.model_dir, 'logs', f'fold_{fold}')
    os.makedirs(log_dir, exist_ok=True)
    # 移除50%负样本
    y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
    X_train_flat = x_train.reshape(x_train.shape[0], -1)
    # 设置固定随机种子
    np.random.seed(42)
    # 分离正负样本索引
    neg_idx = np.where(y_train_flat == 0)[0]
    pos_idx = np.where(y_train_flat == 1)[0]
    # 随机选择一半负样本
    if len(neg_idx) > 0:
        remove_idx = np.random.choice(neg_idx, size=int(len(neg_idx)*0.5), replace=False)
        keep_neg_idx = np.setdiff1d(neg_idx, remove_idx)
    else:
        keep_neg_idx = np.array([], dtype=int)
    # 保留所有正样本
    keep_idx = np.concatenate([pos_idx, keep_neg_idx])
    X_train_flat = X_train_flat[keep_idx]
    y_train_flat = y_train_flat[keep_idx]
    # 通过SMOTE过采样正样本至1:1.5的正负比例
    n_leads = x_train.shape[2] if len(x_train.shape) == 3 else 1
    # try:
    #     from imblearn.over_sampling import SMOTE
    # except ImportError:
    #     raise ImportError("请安装 imblearn 库 用于SMOTE过采样")
    # # 1:1.5比例 -> 少数类(正样本) / 多数类(负样本) = 2/3
    # sampling_strategy = 2/3
    # smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    # X_res, y_res = smote.fit_resample(X_train_flat, y_train_flat)
    # seq_len = x_train.shape[1]
    # x_train_res = X_res.reshape(-1, seq_len, n_leads)
    # y_train_res = y_res.astype(np.float32)
    if n_leads == 1 and len(x_val.shape) == 2:
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    train_sequence = ECGSequence(x_train, y_train, batch_size=args.batch_size, shuffle=True, use_augmentation=False)
    val_sequence = ECGSequence(x_val, y_val, batch_size=args.batch_size, shuffle=False, scalers=train_sequence.scalers, use_augmentation=False)
    train_sequence.save_scalers(os.path.join(model_dir, 'scalers.npy'))
    model = get_model(max_seq_length=args.max_seq_length, n_classes=1, last_layer='sigmoid', n_leads=n_leads,
                      num_blocks=4, num_filters=64, kernel_size=32)
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss=focal_loss(gamma=2.0, alpha=0.5),
                  metrics=['accuracy',
                           AUC(name='auc_1'),
                           Precision(name='precision_1', thresholds=0.5),
                           Recall(name='recall_1', thresholds=0.5),
                           F1Score(name='f1_score', threshold=0.5)])
    tf.get_logger().setLevel('ERROR')
    callbacks = [
        EarlyStopping(monitor='val_f1_score', patience=20, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_f1_score', factor=0.5, patience=10, mode='max', min_lr=1e-6),
        ModelCheckpoint(os.path.join(model_dir, f'model_best_wavenet_fold_{fold}.hdf5'),
                        monitor='val_f1_score', save_best_only=True, mode='max'),
        CSVLogger(os.path.join(log_dir, f'training_fold_{fold}.csv')),
        ProgressCallback(args.epochs)
    ]
    history = model.fit(train_sequence, validation_data=val_sequence, epochs=args.epochs,
                        callbacks=callbacks, verbose=0)
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train WaveNet model for ECG classification')
    parser.add_argument('--data_dir', type=str, default='data/physionet/training2017', help='Directory containing the ECG data files')
    parser.add_argument('--reference_file', type=str, default='data/REFERENCE-v3.csv', help='Path to reference labels file')
    parser.add_argument('--model_dir', type=str, default='WaveNet', help='Directory to save the trained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--max_seq_length', type=int, default=4096, help='Maximum sequence length for ECG signals')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    args = parser.parse_args()
    # 创建输出目录结构
    for subdir in ['models', 'logs', 'results']:
        os.makedirs(os.path.join(args.model_dir, subdir), exist_ok=True)
    print("加载数据...")
    folds = load_data(args.data_dir, args.reference_file, args.max_seq_length, n_splits=args.n_splits, use_smote=True)

    total_samples = len(folds[0][1]) if isinstance(folds, list) else len(folds[1])
    total_pos_samples = np.sum(folds[0][1]) if isinstance(folds, list) else np.sum(folds[1])
    total_pos_ratio = total_pos_samples / total_samples
    print(f"\n数据集信息:\n总样本数: {total_samples}\n正样本数: {total_pos_samples}\n正样本比例: {total_pos_ratio:.4f}")
    fold_histories = []
    print(f"\n开始 {args.n_splits} 折交叉验证训练...")
    for fold, (x_train, y_train, x_val, y_val) in enumerate(folds, 1):
        print(f"\n训练第 {fold} 折")
        train_pos_ratio = np.mean(y_train); val_pos_ratio = np.mean(y_val)
        print(f"训练集大小: {len(y_train)}, 正样本比例: {train_pos_ratio:.4f}")
        print(f"验证集大小: {len(y_val)}, 正样本比例: {val_pos_ratio:.4f}")
        model, history = train_fold(x_train, y_train, x_val, y_val, fold, args)
        fold_histories.append(history.history)
        print(f"第 {fold} 折训练完成")
    if fold_histories:
        best_f1_scores = [np.max(hist['val_f1_score']) for hist in fold_histories]
        best_auc_scores = [np.max(hist['val_auc_1']) for hist in fold_histories]
        print(f"\n平均最佳 F1: {np.mean(best_f1_scores):.4f}, 平均最佳 AUC: {np.mean(best_auc_scores):.4f}")
    results_dir = os.path.join(args.model_dir, 'results')
    metrics = ['loss', 'accuracy', 'auc_1', 'f1_score', 'val_loss', 'val_accuracy', 'val_auc_1', 'val_f1_score']
    avg_metrics = {metric: [] for metric in metrics}
    for history in fold_histories:
        best_epoch = np.argmax(history['val_f1_score'])
        for metric in metrics:
            if metric in history:
                avg_metrics[metric].append(history[metric][best_epoch])
    summary = {metric: float(np.mean(values)) for metric, values in avg_metrics.items() if values}
    with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print("\n交叉验证平均性能:")
    print(f"验证集 F1 分数: {summary.get('val_f1_score', 0):.4f}")
    print(f"验证集 AUC: {summary.get('val_auc_1', 0):.4f}")
    print(f"验证集准确率: {summary.get('val_accuracy', 0):.4f}")
    print(f"验证集损失: {summary.get('val_loss', 0):.4f}")
    # 绘制训练集和验证集指标曲线（与其他训练文件对齐）
    if fold_histories:
        # 1. 绘制最后一折的训练曲线（与其他训练文件一致）
        last_history = fold_histories[-1]
        metrics = ['precision_1', 'recall_1', 'auc_1', 'f1_score']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(last_history[metric], label=f'Train {metric}')
            plt.plot(last_history[f'val_{metric}'], label=f'Val {metric}')
            plt.title(f'Training vs Validation {metric} over epochs')
            plt.xlabel('Epoch'); plt.ylabel(metric); plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(results_dir, f'{metric}_curve.png'))
            plt.close()
        
        # 2. 额外添加平均性能图表
        # 收集每折最佳epoch对应的指标
        fold_best_metrics = {metric: {'train': [], 'val': []} for metric in metrics}
        for i, history in enumerate(fold_histories, 1):
            best_epoch = np.argmax(history['val_f1_score'])
            for metric in metrics:
                if metric in history:
                    fold_best_metrics[metric]['train'].append(history[metric][best_epoch])
                if f'val_{metric}' in history:
                    fold_best_metrics[metric]['val'].append(history[f'val_{metric}'][best_epoch])
        
        # 为每个指标创建折线平均性能图
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # 绘制每一折的最佳值（散点图）
            for i in range(len(fold_best_metrics[metric]['train'])):
                plt.scatter(i+1, fold_best_metrics[metric]['train'][i], c='blue', s=100, alpha=0.7, 
                           label='Train' if i==0 else "")
                plt.scatter(i+1, fold_best_metrics[metric]['val'][i], c='red', s=100, alpha=0.7,
                           label='Validation' if i==0 else "")
            
            # 计算平均值
            train_avg = np.mean(fold_best_metrics[metric]['train'])
            val_avg = np.mean(fold_best_metrics[metric]['val'])
            
            # 绘制平均值线
            plt.axhline(train_avg, color='blue', linestyle='--', alpha=0.7, 
                       label=f'Avg Train: {train_avg:.4f}')
            plt.axhline(val_avg, color='red', linestyle='--', alpha=0.7,
                       label=f'Avg Val: {val_avg:.4f}')
            
            plt.title(f'Best {metric} Performance Across {len(fold_histories)} Folds')
            plt.xlabel('Fold Number')
            plt.ylabel(metric)
            plt.xticks(range(1, len(fold_histories)+1))
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(results_dir, f'{metric}_avg_performance.png'))
            plt.close()
        
        # 3. 添加折叠训练历史曲线图（显示所有折的学习曲线）
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # 绘制每一折的训练和验证曲线
            for i, history in enumerate(fold_histories):
                if metric in history and f'val_{metric}' in history:
                    epochs = range(1, len(history[metric]) + 1)
                    plt.plot(epochs, history[metric], '--', alpha=0.5, 
                             label=f'Fold {i+1} Train')
                    plt.plot(epochs, history[f'val_{metric}'], '-', alpha=0.5, 
                             label=f'Fold {i+1} Val')
            
            # 找出最长的fold历史记录长度
            max_epochs = max(len(h[metric]) for h in fold_histories if metric in h)
            
            # 计算平均曲线
            train_avgs = np.zeros(max_epochs)
            val_avgs = np.zeros(max_epochs)
            train_counts = np.zeros(max_epochs)
            val_counts = np.zeros(max_epochs)
            
            for history in fold_histories:
                if metric in history:
                    for e, val in enumerate(history[metric]):
                        train_avgs[e] += val
                        train_counts[e] += 1
                
                if f'val_{metric}' in history:
                    for e, val in enumerate(history[f'val_{metric}']):
                        val_avgs[e] += val
                        val_counts[e] += 1
            
            # 计算平均值
            for e in range(max_epochs):
                if train_counts[e] > 0:
                    train_avgs[e] /= train_counts[e]
                if val_counts[e] > 0:
                    val_avgs[e] /= val_counts[e]
            
            # 绘制平均曲线
            epochs = range(1, max_epochs + 1)
            plt.plot(epochs, train_avgs, 'b-', linewidth=2, label='Average Train')
            plt.plot(epochs, val_avgs, 'r-', linewidth=2, label='Average Val')
            
            plt.title(f'Average {metric} Learning Curves Across Folds')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(results_dir, f'{metric}_all_folds.png'))
            plt.close()
