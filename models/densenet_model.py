from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling1D, Dense, Dropout,
    MaxPooling1D, AveragePooling1D
)
from tensorflow.keras.models import Model
import numpy as np

def dense_block(x, blocks, growth_rate):
    """简化版Dense Block"""
    for i in range(blocks):
        x_input = x
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(growth_rate, 3, padding='same', use_bias=False)(x)
        x = Concatenate()([x_input, x])
    return x

def transition_layer(x, reduction):
    """简化版Transition Layer"""
    channels = int(x.shape[-1] * reduction)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(channels, 1, padding='same', use_bias=False)(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    return x

def get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid', n_leads=12):
    """
    创建简化版DenseNet模型
    """
    growth_rate = 16
    blocks = [4, 6, 8]
    reduction = 0.5
    
    # 输入层
    inputs = Input(shape=(max_seq_length, n_leads), dtype=np.float32, name='signal')
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
            x = Dropout(0.2)(x)  # 在转换层后添加Dropout防止过拟合
    
    # 分类层
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation=last_layer)(x)
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = get_model()
    model.summary()
