import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
import os
import random
import glob
import wfdb
from scipy.signal import resample
from tqdm import tqdm

# 加载数据集时使用的全局标准化器
GLOBAL_SCALER = None

def normalize_dataset(data, fit_scaler=True, scaler=None):
    """
    对数据集进行标准化处理
    
    参数:
        data: 形状为(n_samples, seq_length)或(n_samples, seq_length, n_leads)的数据
        fit_scaler: 是否拟合新的标准化器
        scaler: 预先拟合好的标准化器，如果提供则使用它
        
    返回:
        normalized_data: 标准化后的数据
        scaler: 使用的标准化器
    """
    global GLOBAL_SCALER
    
    # 保存原始形状
    original_shape = data.shape
    n_dims = len(original_shape)
    
    # 重塑数据以便于标准化处理
    if n_dims == 3:  # (n_samples, seq_length, n_leads)
        n_samples, seq_length, n_leads = original_shape
        reshaped_data = data.reshape(-1, n_leads)
    else:  # (n_samples, seq_length)
        reshaped_data = data.reshape(-1, 1)
    
    # 根据参数选择或创建标准化器
    if scaler is not None:
        # 使用传入的标准化器
        norm_data = scaler.transform(reshaped_data)
    elif GLOBAL_SCALER is not None and not fit_scaler:
        # 使用全局标准化器
        norm_data = GLOBAL_SCALER.transform(reshaped_data)
        scaler = GLOBAL_SCALER
    else:
        # 创建并拟合新的标准化器
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(reshaped_data)
        if fit_scaler:
            GLOBAL_SCALER = scaler
    
    # 将数据重塑回原始形状
    if n_dims == 3:
        normalized_data = norm_data.reshape(original_shape)
    else:
        normalized_data = norm_data.reshape(original_shape)
    
    return normalized_data, scaler

class ECGSequence(Sequence):
    """ECG数据生成器"""
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, scalers=None, use_augmentation=False):
        """
        初始化数据生成器
        
        Args:
            x_data: ECG数据
            y_data: 标签
            batch_size: 批次大小
            shuffle: 是否打乱数据
            scalers: 标准化参数
            use_augmentation: 是否使用数据增强
        """
        # 确保数据类型为float32
        self.x_data = x_data.astype(np.float32)
        self.y_data = y_data.astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        
        # 初始化或使用已有的标准化器
        if scalers is None:
            if len(x_data.shape) == 3:  # 多导联数据
                self.scalers = [StandardScaler() for _ in range(x_data.shape[2])]
                for i in range(x_data.shape[2]):
                    self.scalers[i].fit(self.x_data[:, :, i].reshape(-1, 1))
            else:  # 单导联数据 (time_steps,)
                self.scalers = [StandardScaler()]
                self.scalers[0].fit(self.x_data.reshape(-1, 1))
        else:
            self.scalers = scalers
            
        # 标准化数据
        if len(self.x_data.shape) == 3:  # 多导联数据 (batch, time_steps, leads)
            self.x_data_normalized = np.zeros_like(self.x_data, dtype=np.float32)
            for i in range(self.x_data.shape[2]):
                self.x_data_normalized[:, :, i] = self.scalers[i].transform(
                    self.x_data[:, :, i].reshape(-1, 1)).reshape(self.x_data.shape[0], -1)
        else:  # 单导联数据 (batch, time_steps)
            self.x_data_normalized = self.scalers[0].transform(
                self.x_data.reshape(-1, 1)).reshape(self.x_data.shape)
            
        self.indexes = np.arange(len(self.x_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """返回批次数"""
        return int(np.ceil(len(self.x_data) / self.batch_size))
    
    def __getitem__(self, idx):
        """获取一个批次的数据"""
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = self.x_data_normalized[batch_indexes].astype(np.float32)
        y_batch = self.y_data[batch_indexes].astype(np.float32)
        
        if self.use_augmentation:
            x_batch = self._augment_batch(x_batch)
        
        # 确保单导联数据有正确的形状：(batch_size, time_steps, 1)
        if len(x_batch.shape) == 2:  # 如果是(batch_size, time_steps)形状
            x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], 1)
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        """每个epoch结束时打乱数据"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def save_scalers(self, path):
        """保存标准化参数"""
        np.save(path, self.scalers)
    
    @staticmethod
    def load_scalers(path):
        """加载标准化参数"""
        return np.load(path, allow_pickle=True)
    
    def _augment_batch(self, x_batch):
        """对批次数据进行增强"""
        augmented_batch = x_batch.copy()
        
        # 判断数据维度
        is_multi_lead = len(augmented_batch.shape) == 3
        
        for i in range(len(augmented_batch)):
            if random.random() < 0.3:  # 降低增强概率到30%
                # 添加轻微的高斯噪声
                if is_multi_lead:
                    noise = np.random.normal(0, 0.005, augmented_batch[i].shape).astype(np.float32)
                else:
                    noise = np.random.normal(0, 0.005, augmented_batch[i].shape).astype(np.float32)
                augmented_batch[i] = augmented_batch[i] + noise
                
                # 轻微的时间偏移
                shift = np.random.randint(-2, 3)
                if shift != 0:
                    if is_multi_lead:
                        augmented_batch[i] = np.roll(augmented_batch[i], shift, axis=0)
                    else:
                        augmented_batch[i] = np.roll(augmented_batch[i], shift)
                    
                # 轻微的振幅缩放
                scale = np.random.uniform(0.99, 1.01)
                augmented_batch[i] = augmented_batch[i] * scale
                
        return augmented_batch.astype(np.float32)

class ECGPredictSequence(Sequence):
    """ECG预测数据生成器，仅用于模型推理阶段"""
    def __init__(self, data, batch_size=32, scalers=None):
        """
        初始化预测数据生成器
        
        Args:
            data: 心电图数据
            batch_size: 批次大小
            scalers: 标准化参数
        """
        self.data = data
        self.batch_size = batch_size
        self.scalers = scalers
        
        # 创建索引数组
        self.indexes = np.arange(len(data))
        
        # 如果提供了标准化参数，应用它们
        if self.scalers is not None:
            if len(data.shape) == 3:  # 多导联数据
                n_leads = data.shape[2]  # 获取导联数
                for i in range(n_leads):
                    self.data[:, :, i] = self.scalers[i].transform(self.data[:, :, i])
            else:  # 单导联数据
                self.data = self.scalers[0].transform(self.data.reshape(-1, 1)).reshape(self.data.shape)
    
    def __len__(self):
        """返回批次数"""
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        """获取一个批次的数据"""
        # 获取当前批次的索引
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # 获取当前批次的数据
        batch_data = self.data[batch_indexes]
        
        # 确保数据类型正确
        batch_data = batch_data.astype(np.float32)
        
        # 确保单导联数据有正确的形状：(batch_size, time_steps, 1)
        if len(batch_data.shape) == 2:  # 如果是(batch_size, time_steps)形状
            batch_data = batch_data.reshape(batch_data.shape[0], batch_data.shape[1], 1)
        
        return batch_data  # 只返回数据，不返回标签
    
    @staticmethod
    def load_scalers(path):
        """加载标准化参数"""
        return np.load(path, allow_pickle=True)

def load_data(data_dir, reference_file, max_seq_length=None, normalize=False, n_splits=5, random_state=42):
    """
    加载PhysioNet挑战数据集，并可选进行K折交叉验证划分
    
    参数:
        data_dir: 数据目录路径
        reference_file: 标签文件路径
        max_seq_length: 最大序列长度，如果提供则将所有序列调整为此长度
        normalize: 是否对数据进行标准化
        n_splits: K折交叉验证的折数（默认5折）
        random_state: 随机种子
    
    返回:
        如果n_splits > 1: folds列表，每个元素为(x_train, y_train, x_val, y_val)
        如果n_splits <= 1: (x_data, y_data) 元组
    """
    print(f"正在从 {reference_file} 加载标签文件...")
    ref_df = pd.read_csv(reference_file, header=None, names=['filename', 'label'])
    print(f"标签文件加载完成，共 {len(ref_df)} 条记录")
    
    # 将标签映射到二元标签（房颤为1，其他为0）
    ref_df['af_label'] = ref_df['label'].apply(lambda x: 1 if x == 'A' else 0)
    
    # 加载数据
    data_list = []
    labels = []
    count = 0
    
    for i, row in ref_df.iterrows():
        if count % 100 == 0:
            print(f"已处理 {count}/{len(ref_df)} 条记录")
        
        count += 1
        filename = row['filename']
        label = row['af_label']
        
        try:
            # 构建文件路径
            file_path = os.path.join(data_dir, filename)
            
            # 读取数据
            record = wfdb.rdrecord(file_path)
            signals = record.p_signal.T  # 转置使形状为(n_leads, seq_length)
            
            # 重采样到指定长度
            if max_seq_length is not None and signals.shape[1] != max_seq_length:
                resampled_signals = np.zeros((signals.shape[0], max_seq_length))
                for lead_idx in range(signals.shape[0]):
                    resampled_signals[lead_idx] = resample(signals[lead_idx], max_seq_length)
                signals = resampled_signals
            
            # 转置回(seq_length, n_leads)形式
            signals = signals.T
            
            data_list.append(signals)
            labels.append(label)
        except Exception as e:
            print(f"加载文件 {filename} 时出错: {e}")
    
    # 将列表转换为NumPy数组
    x_data = np.array(data_list)
    y_data = np.array(labels)
    
    print("\n数据加载完成:")
    print(f"ECG数据形状: {x_data.shape}")
    print(f"成功加载的样本数: {len(x_data)}")
    print(f"正样本数（房颤）: {np.sum(y_data == 1)}")
    print(f"负样本数（正常/其他）: {np.sum(y_data == 0)}")
    print(f"正样本比例: {np.mean(y_data):.4f}")
    
    # 如果不进行交叉验证，则直接返回
    if n_splits is None or n_splits <= 1:
        # 标准化数据（可选）
        if normalize:
            print("正在标准化训练数据...")
            x_data, _ = normalize_dataset(x_data, fit_scaler=True)
        return x_data, y_data
    
    # 否则进行K折划分
    print(f"\n正在使用K折交叉验证加载数据 (n_splits={n_splits})...")
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    fold_idx = 1
    
    for train_idx, val_idx in kfold.split(x_data, y_data):
        print(f"\n处理第 {fold_idx}/{n_splits} 折...")
        
        x_train_fold, y_train_fold = x_data[train_idx], y_data[train_idx]
        x_val_fold, y_val_fold = x_data[val_idx], y_data[val_idx]
        
        # 如需标准化，每折独立进行
        if normalize:
            print(f"正在标准化第 {fold_idx} 折数据...")
            global GLOBAL_SCALER
            GLOBAL_SCALER = None
            x_train_fold, _ = normalize_dataset(x_train_fold, fit_scaler=True)
            x_val_fold, _ = normalize_dataset(x_val_fold, fit_scaler=False)
        
        # 打印当前折的数据统计
        print(f"第 {fold_idx} 折训练集: {len(x_train_fold)} 样本, 形状: {x_train_fold.shape}")
        print(f"第 {fold_idx} 折验证集: {len(x_val_fold)} 样本, 形状: {x_val_fold.shape}")
        print(f"第 {fold_idx} 折训练集正例比例: {np.mean(y_train_fold):.4f}")
        print(f"第 {fold_idx} 折验证集正例比例: {np.mean(y_val_fold):.4f}")
        
        folds.append((x_train_fold, y_train_fold, x_val_fold, y_val_fold))
        fold_idx += 1
    
    print(f"\nK折交叉验证数据准备完成，共 {n_splits} 折")
    return folds

from imblearn.over_sampling import SMOTE

def load_data(data_dir, reference_file, max_seq_length=None, normalize=False, 
              n_splits=5, random_state=42, use_smote=True):
    """
    加载PhysioNet挑战数据集，并可选进行K折交叉验证划分
    
    参数:
        data_dir: 数据目录路径
        reference_file: 标签文件路径
        max_seq_length: 最大序列长度
        normalize: 是否对数据进行标准化
        n_splits: K折交叉验证的折数
        random_state: 随机种子
        use_smote: 是否使用SMOTE平衡类别
    """
    print(f"正在从 {reference_file} 加载标签文件...")
    ref_df = pd.read_csv(reference_file, header=None, names=['filename', 'label'])
    print(f"标签文件加载完成，共 {len(ref_df)} 条记录")
    
    # 将标签映射到二元标签（房颤为1，其他为0）
    ref_df['af_label'] = ref_df['label'].apply(lambda x: 1 if x == 'A' else 0)
    
    # 加载数据
    data_list = []
    labels = []
    
    for i, row in tqdm(ref_df.iterrows(), total=len(ref_df)):
        filename = row['filename']
        label = row['af_label']
        
        try:
            file_path = os.path.join(data_dir, filename)
            record = wfdb.rdrecord(file_path)
            signals = record.p_signal.T
            
            if max_seq_length is not None and signals.shape[1] != max_seq_length:
                resampled_signals = np.zeros((signals.shape[0], max_seq_length))
                for lead_idx in range(signals.shape[0]):
                    resampled_signals[lead_idx] = resample(signals[lead_idx], max_seq_length)
                signals = resampled_signals
            
            data_list.append(signals.T)  # 保持(seq_length, n_leads)形状
            labels.append(label)
        except Exception as e:
            print(f"加载文件 {filename} 时出错: {e}")
    
    # 转换为NumPy数组
    x_data = np.array(data_list)
    y_data = np.array(labels)
    
    print("\n数据加载完成:")
    print(f"原始数据 - 正样本(房颤): {np.sum(y_data == 1)}, 负样本: {np.sum(y_data == 0)}")
    
    # 应用SMOTE过采样
    if use_smote:
        print("\n应用SMOTE过采样...")
        original_shape = x_data.shape
        n_samples, seq_length, n_leads = original_shape
        
        # 将3D数据reshape为2D (n_samples, seq_length*n_leads)
        x_reshaped = x_data.reshape(n_samples, -1)
        
        sm = SMOTE(random_state=random_state)
        x_resampled, y_resampled = sm.fit_resample(x_reshaped, y_data)
        
        # 恢复原始形状
        x_data = x_resampled.reshape(-1, seq_length, n_leads)
        y_data = y_resampled
        
        print(f"过采样后 - 正样本: {np.sum(y_data == 1)}, 负样本: {np.sum(y_data == 0)}")
    
#     # 标准化处理
#     if normalize:
#         print("\n标准化数据...")
#         x_data, _ = normalize_dataset(x_data, fit_scaler=True)
    
    # 如果不进行交叉验证，直接返回
    if n_splits is None or n_splits <= 1:
        return x_data, y_data
    
    # K折交叉验证划分
    print(f"\n进行K折划分 (n_splits={n_splits})...")
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for train_idx, val_idx in kfold.split(x_data, y_data):
        x_train, y_train = x_data[train_idx], y_data[train_idx]
        x_val, y_val = x_data[val_idx], y_data[val_idx]
        
#         # 验证集不应用SMOTE
#         if use_smote:
#             print("\n对训练集应用SMOTE...")
#             n_samples = x_train.shape[0]
#             x_train_reshaped = x_train.reshape(n_samples, -1)
            
#             sm = SMOTE(random_state=random_state+len(folds))  # 不同折不同随机种子
#             x_train_res, y_train_res = sm.fit_resample(x_train_reshaped, y_train)
            
#             x_train = x_train_res.reshape(-1, seq_length, n_leads)
#             y_train = y_train_res
        
        folds.append((x_train, y_train, x_val, y_val))
    
    print(f"\n数据准备完成，共 {len(folds)} 折")
    return folds


def load_single_record(file_path, max_seq_length=None, normalize=True):
    """
    加载单个ECG记录文件并标准化
    
    参数:
        file_path: 文件路径
        max_seq_length: 最大序列长度
        normalize: 是否标准化
    
    返回:
        normalized_data: 标准化后的ECG数据
    """
    try:
        record = wfdb.rdrecord(file_path)
        signals = record.p_signal.T  # 转置为(n_leads, seq_length)
        
        # 重采样
        if max_seq_length is not None and signals.shape[1] != max_seq_length:
            resampled_signals = np.zeros((signals.shape[0], max_seq_length))
            for lead_idx in range(signals.shape[0]):
                resampled_signals[lead_idx] = resample(signals[lead_idx], max_seq_length)
            signals = resampled_signals
        
        # 转置回(seq_length, n_leads)
        signals = signals.T
        
        # 标准化
        if normalize:
            signals = signals.reshape(1, *signals.shape)  # 添加批次维度
            signals, _ = normalize_dataset(signals, fit_scaler=False)
            signals = signals[0]  # 移除批次维度
        
        return signals
    
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def get_class_weights(y_data, balanced=True):
    """
    计算类别权重
    
    参数:
        y_data: 标签数组
        balanced: 是否使用平衡权重
    
    返回:
        class_weight_dict: 类别权重字典
    """
    if balanced:
        n_samples = len(y_data)
        n_classes = len(np.unique(y_data))
        class_counts = np.bincount(y_data.astype(int))
        weights = n_samples / (n_classes * class_counts)
        class_weight_dict = {i: weights[i] for i in range(len(weights))}
    else:
        class_weight_dict = {0: 1.0, 1: 1.0}
    return class_weight_dict
