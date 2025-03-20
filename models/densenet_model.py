from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, 
    Concatenate, GlobalAveragePooling1D, Dense, Dropout,
    MaxPooling1D, AveragePooling1D)
from tensorflow.keras.models import Model
import numpy as np

def dense_block(x, blocks, growth_rate):
    """简化版密集块
    
    参数:
    - blocks: 密集层的数量
    - growth_rate: 增长率（每层的滤波器数量）
    """
    for i in range(blocks):
        # 保存输入特征
        x_input = x
        
        # BN-ReLU-Conv(3x3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(growth_rate, 3, padding='same', use_bias=False)(x)
        
        # 连接前面的层
        x = Concatenate()([x_input, x])
    
    return x

def transition_layer(x, reduction):
    """简化版转换层
    
    参数:
    - reduction: 通道数的压缩比例
    """
    channels = int(x.shape[-1] * reduction)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(channels, 1, padding='same', use_bias=False)(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    return x

def get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid'):
    """
    创建简化版DenseNet模型
    
    参数:
    - max_seq_length: 输入序列的最大长度，默认为4096
    - n_classes: 输出类别数，默认为1（二分类）
    - last_layer: 最后一层的激活函数，默认为'sigmoid'
    """
    # 简化的模型参数
    growth_rate = 16  # 减少特征增长率
    blocks = [4, 6, 8]  # 减少密集块的层数
    reduction = 0.5
    
    # 输入层
    inputs = Input(shape=(max_seq_length, 12), dtype=np.float32, name='signal')
    
    # 初始卷积层
    x = Conv1D(32, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # 密集块和转换层
    for i, block_size in enumerate(blocks):
        x = dense_block(x, block_size, growth_rate)
        if i != len(blocks) - 1:
            x = transition_layer(x, reduction)
    
    # 分类层
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    
    # 简化的全连接层
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    outputs = Dense(n_classes, activation=last_layer)(x)
    
    # 创建模型
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # 测试模型
    model = get_model()
    model.summary() 