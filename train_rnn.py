import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.metrics import AUC, Precision, Recall
from tqdm import tqdm
from datasets import ECGSequence, load_data, get_class_weights
from models.rnn_model import get_model

# 设置随机种子以提高可重复性
tf.random.set_seed(42)
np.random.seed(42)

def focal_loss(gamma=1.5, alpha=0.25):
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
    def __init__(self, total_epochs, train_sequence):
        super().__init__()
        self.total_epochs = total_epochs
        self.train_sequence = train_sequence
        self.epoch_pbar = None
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")
        n_batches = len(self.train_sequence)
        self.epoch_pbar = tqdm(total=n_batches, 
                             position=0, 
                             leave=True,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - loss: {postfix}')
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        train_precision = logs.get('precision_1', 0)
        train_recall = logs.get('recall_1', 0)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        metrics_str = f"loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc_1', 0):.4f} - f1: {train_f1:.4f}"
        self.epoch_pbar.set_postfix_str(metrics_str)
        self.epoch_pbar.update(1)
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_pbar:
            self.epoch_pbar.close()
            print()

def train_fold(x_train, y_train, x_val, y_val, fold, args):
    """训练单个fold的模型"""
    print(f"\n开始训练 fold {fold}")
    
    # 创建输出目录
    model_dir = os.path.join(args.model_dir, 'models', f'fold_{fold}')
    log_dir = os.path.join(args.model_dir, 'logs', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 计算类别权重
    class_weight_dict = get_class_weights(y_train.flatten())
    
    # 创建数据生成器（不使用内部标准化以保证一致）
    train_sequence = ECGSequence(
        x_train, y_train,
        batch_size=args.batch_size,
        shuffle=True,
        scalers=None,
        use_augmentation=True
    )
    val_sequence = ECGSequence(
        x_val, y_val,
        batch_size=args.batch_size,
        shuffle=False,
        scalers=train_sequence.scalers,
        use_augmentation=False
    )
    
    # 保存标准化参数
    train_sequence.save_scalers(os.path.join(model_dir, 'scalers.npy'))
    
    # 确定导联数并调整数据形状
    n_leads = 1 if len(x_train.shape) == 2 else x_train.shape[2]
    if len(x_train.shape) == 2:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    else:
        if len(x_val.shape) == 2:
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    
    # 创建模型
    model = get_model(
        max_seq_length=args.max_seq_length,
        n_classes=1,
        last_layer='sigmoid',
        n_leads=n_leads,
        model_type=args.model_type
    )
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', AUC(name='auc_1'), Precision(name='precision_1', thresholds=0.1), Recall(name='recall_1', thresholds=0.1), f1_score]
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
            os.path.join(model_dir, f'model_best_rnn_fold_{fold}.hdf5'),
            monitor='val_f1_score',
            save_best_only=True,
            mode='max'
        ),
        ProgressCallback(args.epochs, train_sequence)
    ]
    
    # 训练模型
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0
    )
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='训练RNN模型，使用PhysioNet数据集进行K折交叉验证')
    parser.add_argument('--data_dir', type=str, default='data/physionet/training2017', help='训练数据目录路径')
    parser.add_argument('--reference_file', type=str, default='data/REFERENCE-v3.csv', help='训练数据标签文件路径')
    parser.add_argument('--model_dir', type=str, default='RNN', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--max_seq_length', type=int, default=4096, help='最大序列长度')
    parser.add_argument('--model_type', type=str, default='gru', choices=['gru', 'lstm'], help='模型类型: gru 或 lstm')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    args = parser.parse_args()
    
    # 创建输出目录
    for subdir in ['models', 'logs', 'results']:
        os.makedirs(os.path.join(args.model_dir, subdir), exist_ok=True)
    
    print("开始加载数据...")
    folds = load_data(args.data_dir, args.reference_file, args.max_seq_length, normalize=True, n_splits=args.n_splits)
    print(f"数据加载完成: 共 {len(folds[0][1]) if isinstance(folds, list) else len(folds[1])} 条训练数据")
    
    fold_results = []
    print(f"\n开始 {args.n_splits} 折交叉验证训练...")
    for fold, (x_train, y_train, x_val, y_val) in enumerate(folds, 1):
        print(f"\n训练第 {fold} 折")
        train_pos_ratio = np.mean(y_train)
        val_pos_ratio = np.mean(y_val)
        print(f"\n第 {fold} 折数据分布:")
        print(f"训练集大小: {len(y_train)}")
        print(f"训练集正样本比例: {train_pos_ratio:.4f}")
        print(f"验证集大小: {len(y_val)}")
        print(f"验证集正样本比例: {val_pos_ratio:.4f}")
        model, history = train_fold(x_train, y_train, x_val, y_val, fold, args)
        fold_results.append({
            'fold': fold,
            'best_val_f1': np.max(history.history.get('val_f1_score', [0])),
            'best_val_auc': np.max(history.history.get('val_auc_1', [0]))
        })
        print(f"第 {fold} 折训练完成，最佳F1: {fold_results[-1]['best_val_f1']:.4f}, 最佳AUC: {fold_results[-1]['best_val_auc']:.4f}")
    
    # 汇总结果
    if fold_results:
        avg_best_f1 = np.mean([res['best_val_f1'] for res in fold_results])
        avg_best_auc = np.mean([res['best_val_auc'] for res in fold_results])
        print(f"\n交叉验证完成，平均最佳 F1: {avg_best_f1:.4f}, 平均最佳 AUC: {avg_best_auc:.4f}")

if __name__ == "__main__":
    main()
