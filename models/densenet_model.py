from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, 
    Concatenate, GlobalAveragePooling1D, Dense, Dropout)
from tensorflow.keras.models import Model
import numpy as np

def dense_block(x, blocks, growth_rate):
    """密集块
    
    参数:
    - blocks: 密集层的数量
    - growth_rate: 增长率（每层的滤波器数量）
    """
    for i in range(blocks):
        # BN-ReLU-Conv
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(growth_rate, 3, padding='same', use_bias=False)(x1)
        
        # 连接前面的所有层
        x = Concatenate()([x, x1])
    return x

def transition_layer(x, reduction):
    """转换层：降低特征图大小和通道数
    
    参数:
    - reduction: 通道数的压缩比例
    """
    channels = int(x.shape[-1] * reduction)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(channels, 1, padding='same', use_bias=False)(x)
    return x

def get_model(n_classes=1, last_layer='sigmoid'):
    """
    创建DenseNet模型用于心电图分类
    
    参数:
    - n_classes: 输出类别数，默认为1（二分类）
    - last_layer: 最后一层的激活函数，默认为'sigmoid'
    
    返回:
    - Keras模型实例
    """
    # 模型参数
    growth_rate = 32  # 每个密集层增加的特征数
    blocks = [6, 12, 24, 16]  # 每个密集块中的层数
    reduction = 0.5  # 转换层的压缩比例
    
    # 输入层
    inputs = Input(shape=(4096, 12), dtype=np.float32, name='signal')
    
    # 初始卷积
    x = Conv1D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 密集块和转换层
    for i, block_size in enumerate(blocks):
        # 密集块
        x = dense_block(x, block_size, growth_rate)
        
        # 转换层（最后一个块不需要）
        if i != len(blocks) - 1:
            x = transition_layer(x, reduction)
    
    # 分类层
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    
    # 输出层
    outputs = Dense(n_classes, activation=last_layer)(x)
    
    # 创建模型
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # 测试模型
    model = get_model()
    model.summary() 