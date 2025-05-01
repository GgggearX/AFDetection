import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, BatchNormalization,
    Dropout, Conv1D, MaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np

def get_model(max_seq_length=1024, n_classes=1, last_layer='sigmoid', n_leads=12):
    """
    构建改进的 CNN-LSTM 模型，增强表达能力并防止过拟合。
    """
    # 输入层
    signal = Input(shape=(max_seq_length, n_leads), dtype=np.float32, name='signal')
    
    # 卷积块 1 (带残差)
    x = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(signal)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    # 残差连接
    skip = Conv1D(filters=32, kernel_size=1, padding='same')(signal)
    skip = BatchNormalization()(skip)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # 卷积块 2 (带残差)
    res = x  # 保存输入用于残差
    x = Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    skip = Conv1D(filters=64, kernel_size=1, padding='same')(res)
    skip = BatchNormalization()(skip)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # 卷积块 3 (带残差)
    res = x
    x = Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    skip = Conv1D(filters=128, kernel_size=1, padding='same')(res)
    skip = BatchNormalization()(skip)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # 卷积块 4 (带残差)
    res = x
    x = Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    skip = Conv1D(filters=128, kernel_size=1, padding='same')(res)
    skip = BatchNormalization()(skip)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # LSTM 部分 - 移除recurrent_dropout参数以启用cuDNN加速
    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False, dropout=0.3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 全连接层
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    output = Dense(n_classes, activation=last_layer)(x)
    return Model(signal, output)
