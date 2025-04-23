import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from datasets import ECGSequence, load_data
from models.transformer_model import get_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
from tqdm import tqdm
from tensorflow.keras import backend as K
import json

def focal_loss(gamma=2.0, alpha=0.25):
    """实现 Focal Loss 损失函数"""
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

def f1_score(y_true, y_pred):
    """计算F1分数，使用0.1作为阈值"""
    y_pred_binary = K.cast(K.greater(y_pred, 0.1), K.floatx())
    true_positives = K.sum(y_true * y_pred_binary)
    possible_positives = K.sum(y_true)
    predicted_positives = K.sum(y_pred_binary)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

class ProgressCallback(tf.keras.callbacks.Callback):
    """训练进度显示回调"""
    def __init__(self, total_epochs):
        super().__init__()
        self.epoch_pbar = None
        self.total_epochs = total_epochs
             
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")
        n_batches = self.params.get('steps', 0)
        self.epoch_pbar = tqdm(total=n_batches, 
                             position=0, 
                             leave=True,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - loss: {postfix}')
             
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        # 计算训练F1分数
        train_precision = logs.get('precision_1', 0)
        train_recall = logs.get('recall_1', 0)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        # 更新进度条和指标
        metrics_str = f"loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc_1', 0):.4f} - f1: {train_f1:.4f}"
        self.epoch_pbar.set_postfix_str(metrics_str)
        self.epoch_pbar.update(1)
             
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_pbar:
            self.epoch_pbar.close()
            print()  # 换行

def train_fold(x_train, y_train, x_val, y_val, fold, args):
    """训练单个fold的模型"""
    print(f"\n开始训练 fold {fold}")
    
    # 创建输出目录
    model_dir = os.path.join(args.model_dir, 'models', f'fold_{fold}')
    log_dir = os.path.join(args.model_dir, 'logs', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建数据生成器
    train_seq = ECGSequence(
        x_train, y_train,
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
    
    # 保存标准化参数
    train_seq.save_scalers(os.path.join(model_dir, 'scalers.npy'))
    
    # 创建模型
    n_leads = x_train.shape[2]
    model = get_model(
        max_seq_length=args.max_seq_length,
        n_classes=1,
        last_layer='sigmoid',
        n_leads=n_leads,
        num_heads=2,
        num_layers=1,
        d_model=64,
        dff=128,
        dropout_rate=0.1
    )
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', 
                AUC(name='auc_1'), 
                Precision(name='precision_1', thresholds=0.1), 
                Recall(name='recall_1', thresholds=0.1),
                f1_score]
    )
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_f1_score',
            patience=20,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.5,
            patience=10,
            mode='max',
            min_lr=1e-6
        ),
        ModelCheckpoint(
            os.path.join(model_dir, f'model_best_transformer_fold_{fold}.hdf5'),
            monitor='val_f1_score',
            save_best_only=True,
            mode='max'
        ),
        CSVLogger(
            os.path.join(log_dir, f'training_fold_{fold}.csv')
        ),
        ProgressCallback(args.epochs)
    ]
    
    # 训练模型
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0
    )
    
    return model, history

def main():
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 设置默认路径
    default_data_dir = 'data/physionet/training2017'
    default_reference_file = 'data/REFERENCE-v3.csv'
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='数据目录路径')
    parser.add_argument('--reference_file', type=str, default=default_reference_file, help='标签文件路径')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--model_dir', type=str, default='Transformer', help='模型保存目录')
    args = parser.parse_args()

    # 创建输出目录
    for subdir in ['models', 'logs', 'results']:
        os.makedirs(os.path.join(args.model_dir, subdir), exist_ok=True)

    print("开始加载数据...")
    # 加载数据
    folds = load_data(args.data_dir, args.reference_file, args.max_seq_length, n_splits=args.n_splits)
    
    # 显示数据集信息
    total_pos_ratio = np.mean(folds[0][1]) if isinstance(folds, list) else np.mean(folds[1])
    total_samples = len(folds[0][1]) if isinstance(folds, list) else len(folds[1])
    total_pos_samples = np.sum(folds[0][1]) if isinstance(folds, list) else np.sum(folds[1])
    print(f"\n数据集信息:")
    print(f"总样本数: {total_samples}")
    print(f"正样本数: {total_pos_samples}")
    print(f"正样本比例: {total_pos_ratio:.4f}")

    # 交叉验证训练
    fold_histories = []
    print(f"\n开始 {args.n_splits} 折交叉验证训练...")
    for fold, (x_train, y_train, x_val, y_val) in enumerate(folds, 1):
        print(f"\n训练第 {fold} 折")
        # 检查当前折的数据分布
        train_pos_ratio = np.mean(y_train)
        val_pos_ratio = np.mean(y_val)
        print(f"\n第 {fold} 折数据分布:")
        print(f"训练集大小: {len(y_train)}")
        print(f"训练集正样本比例: {train_pos_ratio:.4f}")
        print(f"验证集大小: {len(y_val)}")
        print(f"验证集正样本比例: {val_pos_ratio:.4f}")
        model, history = train_fold(x_train, y_train, x_val, y_val, fold, args)
        fold_histories.append(history.history)
        print(f"第 {fold} 折训练完成")

    # 保存交叉验证结果
    results_dir = os.path.join(args.model_dir, 'results')
    metrics = ['loss', 'accuracy', 'auc_1', 'f1_score', 'val_loss', 'val_accuracy', 'val_auc_1', 'val_f1_score']
    avg_metrics = {metric: [] for metric in metrics}
    for history in fold_histories:
        best_epoch = np.argmax(history['val_f1_score'])
        for metric in metrics:
            if metric in history:
                avg_metrics[metric].append(history[metric][best_epoch])
    summary = {metric: np.mean(values) for metric, values in avg_metrics.items() if values}
    with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print("\n交叉验证平均性能:")
    print(f"验证集 F1 分数: {summary.get('val_f1_score', 0):.4f}")
    print(f"验证集 AUC: {summary.get('val_auc_1', 0):.4f}")
    print(f"验证集准确率: {summary.get('val_accuracy', 0):.4f}")
    print(f"验证集损失: {summary.get('val_loss', 0):.4f}")

if __name__ == "__main__":
    main()
