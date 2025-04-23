import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from datasets import ECGSequence, load_data, get_class_weights
from models.wavenet_model import get_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
import argparse
from tqdm import tqdm
from tensorflow.keras import backend as K

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

class F1Score(tf.keras.metrics.Metric):
    """F1分数指标"""
    def __init__(self, name='f1_score', threshold=0.1, **kwargs):
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
    """创建模型保存路径"""
    model_dir = os.path.join(base_dir, 'models', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def train_fold(x_train, y_train, x_val, y_val, fold, args):
    """训练单个fold的模型"""
    print(f"\n开始训练 fold {fold}")
    
    # 创建模型保存路径
    model_dir = create_model_save_path(args.model_dir, fold)
    
    # 计算类别权重
    class_weight_dict = get_class_weights(y_train.flatten())
    
    # 创建数据生成器
    train_sequence = ECGSequence(
        x_train, y_train,
        batch_size=args.batch_size,
        shuffle=True,
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
    scaler_path = os.path.join(model_dir, 'scalers.npy')
    train_sequence.save_scalers(scaler_path)
    
    # 创建模型
    n_leads = x_train.shape[2]
    model = get_model(
        max_seq_length=args.max_seq_length,
        n_classes=1,
        last_layer='sigmoid',
        n_leads=n_leads,
        num_blocks=8,      # WaveNet块的数量
        num_filters=64,    # 每个卷积层的滤波器数量
        kernel_size=32     # 卷积核大小
    )
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', 
                AUC(name='auc_1'), 
                Precision(name='precision_1', thresholds=0.1), 
                Recall(name='recall_1', thresholds=0.1),
                F1Score(name='f1_score', threshold=0.1)]
    )
    
    # 创建自定义回调来显示训练进度
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_pbar = None
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            n_batches = len(train_sequence)
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

    # 禁用 TensorFlow 的默认输出
    tf.get_logger().setLevel('ERROR')
    
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
            os.path.join(model_dir, f'model_best_wavenet_fold_{fold}.hdf5'),
            monitor='val_f1_score',
            save_best_only=True,
            mode='max'
        ),
        ProgressCallback()
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train WaveNet model for ECG classification')
    parser.add_argument('--data_dir', type=str, default='data/physionet/training2017', help='Directory containing the ECG data files')
    parser.add_argument('--reference_file', type=str, default='data/physionet/training2017/REFERENCE.csv', help='Path to the reference file containing labels')
    parser.add_argument('--model_dir', type=str, default='WaveNet', help='Directory to save the trained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length for ECG signals')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    args = parser.parse_args()
    
    # 创建输出目录
    for subdir in ['models', 'logs', 'results']:
        os.makedirs(os.path.join(args.model_dir, subdir), exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    folds = load_data(args.data_dir, args.reference_file, args.max_seq_length, n_splits=args.n_splits)
    print(f"数据加载完成:")
    print(f"成功加载的样本数: {len(folds[0][1]) if isinstance(folds, list) else len(folds[1])}")
    print(f"正样本数（房颤）: {np.sum(folds[0][1] == 1) if isinstance(folds, list) else np.sum(folds[1] == 1)}")
    print(f"负样本数（正常/其他）: {np.sum(folds[0][1] == 0) if isinstance(folds, list) else np.sum(folds[1] == 0)}")
    print(f"正样本比例: {np.mean(folds[0][1]) if isinstance(folds, list) else np.mean(folds[1]):.4f}")
    
    # 初始化K折交叉验证
    fold_results = []
    
    # 训练每个fold
    for fold, (x_train, y_train, x_val, y_val) in enumerate(folds, 1):
        # 检查当前折的数据分布
        train_pos_ratio = np.mean(y_train)
        val_pos_ratio = np.mean(y_val)
        print(f"\n第 {fold} 折数据分布:")
        print(f"训练集大小: {len(y_train)}")
        print(f"训练集正样本比例: {train_pos_ratio:.4f}")
        print(f"验证集大小: {len(y_val)}")
        print(f"验证集正样本比例: {val_pos_ratio:.4f}")
        model, history = train_fold(x_train, y_train, x_val, y_val, fold, args)
        
        # 评估模型
        print(f"\n评估 fold {fold} 的模型性能...")
        val_sequence = ECGSequence(x_val, y_val, batch_size=args.batch_size, shuffle=False, scalers=train_sequence.scalers if 'train_sequence' in locals() else None, use_augmentation=False)
        metrics = model.evaluate(val_sequence, verbose=0)
        fold_results.append({
            'fold': fold,
            'loss': metrics[0],
            'accuracy': metrics[1],
            'auc': metrics[2],
            'precision': metrics[3],
            'recall': metrics[4],
            'f1': metrics[5]
        })
        print(f"\nFold {fold} 结果:")
        print(f"Loss: {metrics[0]:.4f}")
        print(f"Accuracy: {metrics[1]:.4f}")
        print(f"AUC: {metrics[2]:.4f}")
        print(f"Precision: {metrics[3]:.4f}")
        print(f"Recall: {metrics[4]:.4f}")
        print(f"F1 Score: {metrics[5]:.4f}")

    # 计算并输出平均结果
    if fold_results:
        avg_loss = np.mean([r['loss'] for r in fold_results])
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        avg_auc = np.mean([r['auc'] for r in fold_results])
        avg_precision = np.mean([r['precision'] for r in fold_results])
        avg_recall = np.mean([r['recall'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        print(f"\n平均结果: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, AUC={avg_auc:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")

if __name__ == "__main__":
    main()
