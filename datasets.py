import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, x_data, y_data=None, batch_size=8, start_idx=0, end_idx=None):
        """
        初始化序列生成器
        
        参数:
        x_data: 可以是HDF5文件路径(str)或numpy数组
        y_data: 可以是CSV文件路径(str)或numpy数组
        batch_size: 批次大小
        start_idx: 起始索引
        end_idx: 结束索引
        """
        self.batch_size = batch_size
        
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

    @property
    def n_classes(self):
        if self.y is None:
            return 1
        return self.y.shape[1] if len(self.y.shape) > 1 else 1

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        batch_x = np.array(self.x[start:end])
        if self.y is None:
            return batch_x
        batch_y = np.array(self.y[start:end])
        return batch_x, batch_y

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        if self.f is not None and self.close_file:
            self.f.close()
