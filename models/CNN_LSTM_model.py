import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, BatchNormalization, 
    Dropout, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np

def get_model(max_seq_length=1024, n_classes=1, last_layer='sigmoid', n_leads=12):
    """
    创建简化版CNN-LSTM模型
    
    参数:
    - max_seq_length: 输入序列的最大长度，默认为1024
    - n_classes: 输出类别数，默认为1（二分类）
    - last_layer: 最后一层的激活函数，默认为'sigmoid'
    - n_leads: 导联数，默认为12
    """
    # 输入层
    signal = Input(shape=(max_seq_length, n_leads), dtype=np.float32, name='signal')
    
    # CNN部分 - 3个简单的卷积块
    # 第一个卷积块
    x = Conv1D(filters=64, kernel_size=3, padding='same', 
               kernel_regularizer=l2(0.01))(signal)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    
    # 第二个卷积块
    x = Conv1D(filters=128, kernel_size=3, padding='same',
               kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    
    # 第三个卷积块
    x = Conv1D(filters=256, kernel_size=3, padding='same',
               kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    
    # LSTM部分 - 单层LSTM
    x = LSTM(128, return_sequences=True,
             kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = LSTM(64, return_sequences=False,
             kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 全连接层
    x = Dense(32, activation='relu',
              kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 输出层
    output = Dense(n_classes, activation=last_layer)(x)
    
    return Model(signal, output)

if __name__ == "__main__":
    # 测试模型
    model = get_model()
    model.summary() 