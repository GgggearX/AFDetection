import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Bidirectional, GRU, LSTM, Dense,
    Dropout, BatchNormalization, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

def get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid', n_leads=1, model_type='BIGRU'):
    """
    创建用于ECG分类的RNN模型 (支持BiGRU或BiLSTM)
    """
    # 输入形状
    if n_leads == 1:
        input_shape = (max_seq_length, 1)
    else:
        input_shape = (max_seq_length, n_leads)
    inputs = Input(shape=input_shape)
    
    # 初步卷积特征提取和下采样
    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # 添加通道注意力机制（SE模块）以突出重要特征
    se = GlobalAveragePooling1D()(x)
    se = Dense(x.shape[-1] // 16, activation='relu')(se)
    se = Dense(x.shape[-1], activation='sigmoid')(se)
    se = tf.expand_dims(se, axis=1)
    x = x * se
    
    # 双向RNN层
    if model_type.upper() == 'BIGRU':
        x = Bidirectional(GRU(units=128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(units=64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
    else:  # BILSTM
        x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(units=64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
    
    # 全连接层
    x = Dense(units=64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    outputs = Dense(n_classes, activation=last_layer)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid', n_leads=1)
    model.summary()
