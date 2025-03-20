import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        
        # 首先创建训练序列并进行拟合
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train, is_training=True)
        # 使用相同的标准化器创建验证序列
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train, 
                       is_training=False, scalers=train_seq.scalers)
        return train_seq, valid_seq

    def __init__(self, x_data, y_data=None, batch_size=8, start_idx=0, end_idx=None, 
                 is_training=False, scalers=None):
        """
        初始化序列生成器
        
        参数:
        x_data: 可以是HDF5文件路径(str)或numpy数组
        y_data: 可以是CSV文件路径(str)或numpy数组
        batch_size: 批次大小
        start_idx: 起始索引
        end_idx: 结束索引
        is_training: 是否为训练模式（决定是否进行数据增强）
        scalers: 预训练的StandardScaler列表（用于验证/测试集）
        """
        self.batch_size = batch_size
        self.is_training = is_training
        
        # 处理x_data
        if isinstance(x_data, str):
            self.f = h5py.File(x_data, "r")
            self.x = self.f[y_data] if isinstance(y_data, str) else self.f['tracings']
            self.close_file = True
        else:
            self.x = x_data
            self.f = None
            self.close_file = False
            
        # 处理y_data
        if y_data is None:
            self.y = None
        elif isinstance(y_data, str):
            self.y = pd.read_csv(y_data).values
        else:
            self.y = y_data
            
        # 设置索引范围
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

        # 处理标准化器
        if scalers is not None:
            self.scalers = scalers
        else:
            self.scalers = [StandardScaler() for _ in range(12)]
            if is_training:
                # 在训练模式下，对每个导联进行拟合
                x_array = np.array(self.x)
                for i in range(12):
                    # 重塑数据以适应StandardScaler
                    lead_data = x_array[:, :, i].reshape(-1, 1)
                    self.scalers[i].fit(lead_data)
            else:
                # 在预测模式下，如果没有提供标准化器，则使用训练数据的统计信息
                x_array = np.array(self.x)
                for i in range(12):
                    lead_data = x_array[:, :, i].reshape(-1, 1)
                    self.scalers[i].mean_ = np.mean(lead_data)
                    self.scalers[i].var_ = np.var(lead_data)
                    self.scalers[i].scale_ = np.sqrt(self.scalers[i].var_)

    def apply_data_augmentation(self, batch_x):
        """应用数据增强技术"""
        augmented_batch = batch_x.copy()
        
        for i in range(len(augmented_batch)):
            # 随机应用以下增强方法
            if np.random.random() < 0.5:
                # 添加高斯噪声
                noise = np.random.normal(0, 0.01, augmented_batch[i].shape)
                augmented_batch[i] = augmented_batch[i] + noise
                
            if np.random.random() < 0.3:
                # 随机时间偏移
                shift = np.random.randint(-50, 50)
                augmented_batch[i] = np.roll(augmented_batch[i], shift, axis=0)
                
            if np.random.random() < 0.3:
                # 随机振幅缩放
                scale = np.random.uniform(0.8, 1.2)
                augmented_batch[i] = augmented_batch[i] * scale
                
            if np.random.random() < 0.2:
                # 随机基线漂移
                t = np.linspace(0, 1, augmented_batch[i].shape[0])
                baseline = 0.1 * np.sin(2 * np.pi * t * np.random.uniform(0.1, 0.5))
                for j in range(augmented_batch[i].shape[1]):
                    augmented_batch[i, :, j] += baseline
        
        return augmented_batch

    def standardize_batch(self, batch_x):
        """对批次数据进行标准化"""
        standardized_batch = np.zeros_like(batch_x)
        for i in range(12):
            # 重塑数据以适应StandardScaler
            lead_data = batch_x[:, :, i].reshape(-1, 1)
            standardized_lead = self.scalers[i].transform(lead_data)
            standardized_batch[:, :, i] = standardized_lead.reshape(batch_x.shape[0], -1)
        return standardized_batch

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        batch_x = np.array(self.x[start:end])
        
        # 应用标准化
        batch_x = self.standardize_batch(batch_x)
        
        # 在训练模式下应用数据增强
        if self.is_training:
            batch_x = self.apply_data_augmentation(batch_x)
        
        if self.y is None:
            return batch_x
            
        batch_y = np.array(self.y[start:end])
        return batch_x, batch_y

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        if self.f is not None and self.close_file:
            self.f.close()

    def save_scalers(self, save_path):
        """保存标准化参数"""
        scaler_params = []
        for scaler in self.scalers:
            params = {
                'mean_': scaler.mean_,
                'var_': scaler.var_,
                'scale_': scaler.scale_
            }
            scaler_params.append(params)
        np.save(save_path, scaler_params)

    @classmethod
    def load_scalers(cls, load_path):
        """加载标准化参数"""
        scaler_params = np.load(load_path, allow_pickle=True)
        scalers = []
        for params in scaler_params:
            scaler = StandardScaler()
            scaler.mean_ = params['mean_']
            scaler.var_ = params['var_']
            scaler.scale_ = params['scale_']
            scalers.append(scaler)
        return scalers
