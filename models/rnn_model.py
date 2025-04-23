#!/usr/bin/env python
# coding: utf-8

"""
RNN模型定义
包含LSTM和GRU网络结构，用于ECG数据的分类
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Bidirectional, GRU, LSTM, Dense, 
    Dropout, BatchNormalization, Conv1D, MaxPooling1D
)
from tensorflow.keras.models import Model

def get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid', n_leads=1, model_type='BIGRU'):
    """
    创建用于ECG分类的RNN模型
    
    参数:
        max_seq_length: 输入序列的最大长度
        n_classes: 输出类别数，二分类为1，多分类为类别数量
        last_layer: 最后一层的激活函数，二分类为'sigmoid'，多分类为'softmax'
        n_leads: 导联数量，单导联为1，多导联为导联数量
        model_type: 模型类型，可选: 'BIGRU', 'BILSTM'
        
    返回:
        构建好的模型
    """
    # 确定输入形状
    if n_leads == 1:
        input_shape = (max_seq_length, 1)  # 单导联
    else:
        input_shape = (max_seq_length, n_leads)  # 多导联
    
    # 创建输入层
    inputs = Input(shape=input_shape)
    
    # 添加卷积层进行初步特征提取和下采样
    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # 根据指定类型选择RNN模型
    if model_type == 'BIGRU':
        # 双向GRU层
        x = Bidirectional(GRU(units=128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(GRU(units=64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
    else:  # BILSTM
        # 双向LSTM层
        x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(units=64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
    
    # 全连接层
    x = Dense(units=64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    if n_classes == 1:
        # 二分类问题
        outputs = Dense(n_classes, activation=last_layer)(x)
    else:
        # 多分类问题
        outputs = Dense(n_classes, activation=last_layer)(x)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

if __name__ == "__main__":
    # 测试代码
    model = get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid', n_leads=1)
    model.summary()
