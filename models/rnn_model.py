from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization, 
    GlobalAveragePooling1D)
from tensorflow.keras.models import Model
import numpy as np

def get_model(n_classes=1, last_layer='sigmoid'):
    """
    创建双向LSTM模型用于心电图分类
    
    参数:
    - n_classes: 输出类别数，默认为1（二分类）
    - last_layer: 最后一层的激活函数，默认为'sigmoid'
    
    返回:
    - Keras模型实例
    """
    # 输入层 - 保持与原模型相同的输入维度
    signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
    
    # 第一个BiLSTM层
    x = Bidirectional(LSTM(128, return_sequences=True))(signal)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 第二个BiLSTM层
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 第三个BiLSTM层
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 全局池化层
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 输出层
    output = Dense(n_classes, activation=last_layer)(x)
    
    # 创建模型
    model = Model(signal, output)
    return model

if __name__ == "__main__":
    # 测试模型
    model = get_model()
    model.summary() 