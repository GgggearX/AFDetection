from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall
from models.rnn_model import get_model
import argparse
from datasets import ECGSequence
import pandas as pd
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tensorflow.keras import backend as K

def focal_loss(gamma=1.5, alpha=0.25):
    """实现 Focal Loss 损失函数
    
    参数:
        gamma: 聚焦参数，控制难例的权重
        alpha: 平衡正负样本的参数
        
    返回:
        focal_loss_fixed: 损失函数
    """
    def focal_loss_fixed(y_true, y_pred):
        # 确保输入为 float32 类型
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 添加平滑处理，确保数值稳定性
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # 计算正样本和负样本的损失
        pt_1 = tf.where(tf.equal(y_true, 1.0), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0.0), y_pred, tf.zeros_like(y_pred))
        
        # 计算 focal loss
        loss_1 = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1 + epsilon)
        loss_0 = -(1.0 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0 + epsilon)
        
        # 合并损失
        loss = tf.reduce_mean(loss_1 + loss_0)
        return loss
    return focal_loss_fixed

def create_model_save_path(base_dir, fold):
    """创建模型保存路径"""
    model_dir = os.path.join(base_dir, 'models', f'fold_{fold}')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def train_fold(x_data, y_data, fold, args):
    """训练单个fold的模型"""
    print(f"\n开始训练 fold {fold}")
    
    # 创建模型保存路径
    model_dir = create_model_save_path(args.model_dir, fold)
    
    # 计算类别权重
    unique_classes = np.unique(y_data)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_data.flatten()
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    # 创建模型
    n_leads = x_data.shape[2]  # 获取导联数
    model = get_model(
        max_seq_length=args.max_seq_length,
        n_classes=1,
        last_layer='sigmoid',
        n_leads=n_leads
    )
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', AUC(), Precision(), Recall()],
        run_eagerly=False  # 禁用即时执行模式
    )
    
    # 创建自定义回调来显示训练进度
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_pbar = None
            self.step_pbar = None
            
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            # 计算每个epoch的batch数量
            n_batches = len(train_sequence)
            self.epoch_pbar = tqdm(total=n_batches, 
                                 position=0, 
                                 leave=True,
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - loss: {postfix}')
            
        def on_batch_end(self, batch, logs=None):
            if logs is None:
                logs = {}
            # 计算训练F1分数
            train_precision = logs.get('precision', 0)
            train_recall = logs.get('recall', 0)
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            
            # 更新进度条和指标
            metrics_str = f"{logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - auc: {logs.get('auc', 0):.4f} - f1: {train_f1:.4f}"
            self.epoch_pbar.set_postfix_str(metrics_str)
            self.epoch_pbar.update(1)
            
        def on_epoch_end(self, epoch, logs=None):
            if self.epoch_pbar:
                self.epoch_pbar.close()
                print()  # 换行
    
    # 禁用 TensorFlow 的默认输出
    tf.get_logger().setLevel('ERROR')
    
    # 训练模型
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ProgressCallback()
        ],
        class_weight=class_weight_dict,
        verbose=0  # 设置为0，使用自定义输出
    )
    
    # 评估模型
    val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(val_sequence, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    
    return model, history, val_sequence, val_loss, val_acc, val_auc, val_f1

def load_data(data_dir, reference_file, max_seq_length=1024):
    """加载PhysioNet数据集，剔除噪声记录
    
    参数:
        data_dir: 数据目录路径
        reference_file: 标签文件路径
        max_seq_length: 最大序列长度
        
    返回:
        x_data: ECG数据
        y_data: 标签 (0: 正常/其他, 1: 房颤)
    """
    print(f"正在从 {reference_file} 加载标签文件...")
    # 读取标签文件
    df = pd.read_csv(reference_file, header=None, names=['record', 'label'])
    print(f"标签文件加载完成，共 {len(df)} 条记录")
    
    # 初始化数据列表
    x_data_list = []
    y_data_list = []
    
    # 采样频率
    fs = 300  # Hz
    
    # 遍历所有记录
    for idx, row in df.iterrows():
        record_name = row['record']
        label = row['label']
        
        # 跳过噪声记录
        if label == '~':
            continue
            
        # 读取.mat文件
        mat_path = os.path.join(data_dir, f"{record_name}.mat")
        if not os.path.exists(mat_path):
            continue
            
        try:
            # 加载.mat文件
            mat_data = loadmat(mat_path)
            
            # 获取ECG数据
            if 'val' in mat_data:
                ecg_data = mat_data['val']
                
                # 检查数据格式并调整
                if ecg_data.shape[0] == 1:  # 单导联数据
                    # 计算最接近的可以被12整除的长度
                    n = ecg_data.shape[1]
                    adjusted_length = (n // 12) * 12
                    if adjusted_length < n:
                        ecg_data = ecg_data[:, :adjusted_length]
                    
                    # 检查信号是否倒置（通过计算信号均值）
                    mean_val = np.mean(ecg_data)
                    if mean_val < 0:
                        ecg_data = -ecg_data
                    
                    # 重塑数据为(12, n)格式（复制单导联数据到12个导联）
                    n_samples = ecg_data.shape[1] // 12
                    ecg_data = ecg_data.reshape(12, n_samples)
                elif ecg_data.shape[0] != 12:
                    continue
                
                # 转置数据以匹配我们的格式 (time_steps, channels)
                ecg_data = ecg_data.T
                
                # 截断或填充序列长度
                if ecg_data.shape[0] > max_seq_length:
                    ecg_data = ecg_data[:max_seq_length]
                else:
                    pad_width = ((0, max_seq_length - ecg_data.shape[0]), (0, 0))
                    ecg_data = np.pad(ecg_data, pad_width, mode='constant')
                    
                # 将标签转换为数值
                if label == 'A':  # 房颤
                    y = 1
                elif label in ['N', 'O']:  # Normal 和 Other
                    y = 0
                else:
                    continue
                    
                x_data_list.append(ecg_data)
                y_data_list.append(y)
                
                if (idx + 1) % 100 == 0:  # 每处理100条记录打印一次进度
                    print(f"已处理 {idx + 1}/{len(df)} 条记录")
            
        except Exception as e:
            continue
    
    # 转换为numpy数组
    x_data = np.array(x_data_list)
    y_data = np.array(y_data_list).reshape(-1, 1)
    
    print(f"\n数据加载完成:")
    print(f"成功加载的样本数: {len(x_data)}")
    print(f"正样本数（房颤）: {np.sum(y_data == 1)}")
    print(f"负样本数（正常/其他）: {np.sum(y_data == 0)}")
    print(f"正样本比例: {np.mean(y_data == 1):.4f}")
    
    if len(x_data) == 0:
        raise ValueError("没有成功加载任何数据！请检查数据路径和文件格式是否正确。")
    
    return x_data, y_data

def main():
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 设置默认路径
    default_data_dir = 'data/physionet/training2017'
    default_reference_file = 'data/REFERENCE-v3.csv'
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练RNN模型')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='数据目录路径')
    parser.add_argument('--reference_file', type=str, default=default_reference_file, help='标签文件路径')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--max_seq_length', type=int, default=4096, help='最大序列长度')
    parser.add_argument('--model_dir', type=str, default='models/rnn', help='模型保存目录')
    args = parser.parse_args()
    
    # 加载数据
    print("正在加载数据...")
    x_data, y_data = load_data(args.data_dir, args.reference_file, max_seq_length=args.max_seq_length)
    
    # 创建输出目录
    output_dir = 'models/rnn'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置交叉验证
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    # 存储每个折的结果
    fold_results = []
    
    # 对每个折进行训练
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data, y_data), 1):
        print(f"\n开始训练第 {fold} 折...")
        
        # 训练模型
        model, history, val_sequence, val_loss, val_acc, val_auc, val_f1 = train_fold(x_data, y_data, fold, args)
        
        # 保存模型
        model.save(os.path.join(output_dir, f'rnn_model_fold_{fold}.h5'))
        
        fold_results.append({
            'fold': fold,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1
        })
        
        print(f"\n第 {fold} 折结果 - F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
    
    # 打印所有折的平均结果
    avg_results = {
        'val_loss': np.mean([r['val_loss'] for r in fold_results]),
        'val_acc': np.mean([r['val_acc'] for r in fold_results]),
        'val_auc': np.mean([r['val_auc'] for r in fold_results]),
        'val_f1': np.mean([r['val_f1'] for r in fold_results])
    }
    
    print("\n最终结果:")
    print(f"F1: {avg_results['val_f1']:.4f}")
    print(f"AUC: {avg_results['val_auc']:.4f}")
    print(f"Acc: {avg_results['val_acc']:.4f}")

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

if __name__ == '__main__':
    main()

# 执行命令：
# $env:TF_FORCE_GPU_ALLOW_GROWTH='true'
# $env:TF_GPU_ALLOCATOR='cuda_malloc_async'
# python train_rnn.py  # 使用默认参数
# 或者指定参数：
# python train_rnn.py --data_dir 自定义数据目录 --reference_file 自定义标签文件 --epochs 200 --batch_size 64