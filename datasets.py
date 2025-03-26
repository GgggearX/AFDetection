import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.io import loadmat
import os
import random


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
            self.scalers = [StandardScaler() for _ in range(x_data.shape[2])]
            for i in range(x_data.shape[2]):
                self.scalers[i].fit(self.x_data[:, :, i].reshape(-1, 1))
        else:
            self.scalers = scalers
            
        # 标准化数据
        self.x_data_normalized = np.zeros_like(self.x_data, dtype=np.float32)
        for i in range(self.x_data.shape[2]):
            self.x_data_normalized[:, :, i] = self.scalers[i].transform(
                self.x_data[:, :, i].reshape(-1, 1)).reshape(self.x_data.shape[0], -1)
            
        self.indexes = np.arange(len(self.x_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        # 计算类别权重
        unique_classes = np.unique(self.y_data)
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=self.y_data.flatten()
        )
    
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
        """对批次数据进行增强
        
        Args:
            x_batch: 输入批次数据，形状为(batch_size, time_steps, channels)
            
        Returns:
            增强后的数据
        """
        augmented_batch = x_batch.copy()
        for i in range(len(augmented_batch)):
            if random.random() < 0.5:  # 50%的概率进行数据增强
                # 添加高斯噪声
                noise = np.random.normal(0, 0.01, augmented_batch[i].shape).astype(np.float32)
                augmented_batch[i] = augmented_batch[i] + noise
                
                # 时间偏移
                shift = np.random.randint(-3, 4)
                if shift != 0:
                    augmented_batch[i] = np.roll(augmented_batch[i], shift, axis=0)
                    
                # 振幅缩放
                scale = np.random.uniform(0.98, 1.02)
                augmented_batch[i] = augmented_batch[i] * scale
                
                # 基线漂移
                drift = np.linspace(0, np.random.uniform(-0.02, 0.02), augmented_batch[i].shape[0])
                drift = drift.reshape(-1, 1).astype(np.float32)
                augmented_batch[i] = augmented_batch[i] + drift
                
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
            n_leads = data.shape[2]  # 获取导联数
            for i in range(n_leads):
                self.data[:, :, i] = self.scalers[i].transform(self.data[:, :, i])
    
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
            
        return batch_data  # 只返回数据，不返回标签
    
    @staticmethod
    def load_scalers(path):
        """加载标准化参数"""
        return np.load(path, allow_pickle=True)


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
            print(f"警告: 找不到文件 {mat_path}")
            continue
            
        try:
            # 加载.mat文件
            mat_data = loadmat(mat_path)
            
            # 获取ECG数据
            if 'val' in mat_data:
                ecg_data = mat_data['val']
                print(f"\n处理文件 {record_name}:")
                print(f"ECG数据形状: {ecg_data.shape}")
                
                # 检查数据格式并调整
                n_leads = ecg_data.shape[0]  # 获取导联数
                print(f"导联数: {n_leads}")
                
                # 转置数据以匹配我们的格式 (time_steps, channels)
                ecg_data = ecg_data.T
                print(f"转置后的数据形状: {ecg_data.shape}")
                
                # 截断或填充序列长度
                if ecg_data.shape[0] > max_seq_length:
                    ecg_data = ecg_data[:max_seq_length]
                    print(f"截断后的数据形状: {ecg_data.shape}")
                else:
                    pad_width = ((0, max_seq_length - ecg_data.shape[0]), (0, 0))
                    ecg_data = np.pad(ecg_data, pad_width, mode='constant')
                    print(f"填充后的数据形状: {ecg_data.shape}")
                    
                # 将标签转换为数值
                if label == 'A':  # 房颤
                    y = 1
                elif label in ['N', 'O']:  # Normal 和 Other
                    y = 0
                else:
                    print(f"警告: 未知标签 {label}，跳过")
                    continue
                    
                x_data_list.append(ecg_data)
                y_data_list.append(y)
                
                if (idx + 1) % 100 == 0:  # 每处理100条记录打印一次进度
                    print(f"已处理 {idx + 1}/{len(df)} 条记录")
            else:
                print(f"警告: {record_name} 中没有找到'val'字段")
            
        except Exception as e:
            print(f"错误: 处理 {record_name} 时出错: {str(e)}")
            continue
    
    # 转换为numpy数组
    x_data = np.array(x_data_list)
    y_data = np.array(y_data_list).reshape(-1, 1)
    
    print(f"\n数据加载完成:")
    print(f"成功加载的样本数: {len(x_data)}")
    print(f"正样本数（房颤）: {np.sum(y_data == 1)}")
    print(f"负样本数（正常/其他）: {np.sum(y_data == 0)}")
    print(f"正样本比例: {np.mean(y_data == 1):.4f}")
    print(f"跳过的样本数: {len(df) - len(x_data)}")
    
    if len(x_data) == 0:
        raise ValueError("没有成功加载任何数据！请检查数据路径和文件格式是否正确。")
    
    return x_data, y_data
