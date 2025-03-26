import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization, 
    GlobalAveragePooling1D, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization, 
    Add, AveragePooling1D, Conv1D, Activation, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np

def get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid'):
    """
    在原有基础上进行了深度和功能的增强版本，用于心电信号分类或其他长序列任务。
    主要变化和改进：
    1) 在网络前端增加卷积层，用来提取局部特征并减少BiLSTM的输入长度，从而减轻计算负担。
    2) 引入双分支全局池化(平均池化 + 最大池化)后再拼接，加大了对不同统计特征的学习。
    3) 对注意力模块增加前馈网络(FFN)和残差结构，使其更贴近TransformerEncoder的做法。
    4) 增加L2正则化与适度的Dropout，抑制过拟合。
    5) 分层加深网络：三层BiLSTM + 两次Attention模块，整体提升对序列全局和上下文的捕捉能力。
    
    参数:
    - max_seq_length: 输入序列的最大长度，默认为4096
    - n_classes: 输出类别数，默认为1 (二分类)
    - last_layer: 最后一层的激活函数，默认为'sigmoid'
    
    返回:
    - Keras模型实例 (tf.keras.Model)
    """
    
    #-----------------------------
    # 1. 输入层
    #-----------------------------
    signal = Input(shape=(max_seq_length, 12), dtype=np.float32, name='signal')

    #-----------------------------
    # 2. 前置卷积与初步下采样
    #   通过卷积提取局部模式，并用AveragePooling进一步压缩序列长度
    #-----------------------------
    x = Conv1D(
        filters=32,              # 初始滤波器数量
        kernel_size=7,           # 卷积核大小为7，对心电节律等特征有更长的感受野
        strides=1, 
        padding='same',
        kernel_regularizer=regularizers.l2(0.001), # 在卷积层增加L2正则化
        name='conv_initial'
    )(signal)
    x = BatchNormalization(name='bn_initial')(x)
    x = Activation('relu', name='act_initial')(x)
    x = AveragePooling1D(pool_size=2, name='pool_initial')(x)
    x = Dropout(0.1, name='dropout_initial')(x)

    #-----------------------------
    # 3. 第一个双向LSTM层
    #   让模型感知局部时序信息
    #-----------------------------
    x = Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        name='bilstm_1'
    )(x)
    x = BatchNormalization(name='bn_lstm1')(x)
    x = Dropout(0.2, name='dropout_lstm1')(x)

    #-----------------------------
    # 4. 第一次多头注意力(TransformerEncoder式)
    #   + 残差连接 + 前馈网络FFN + LayerNorm
    #-----------------------------
    # 先用AveragePooling进一步压缩序列长度
    x = AveragePooling1D(pool_size=2, name='pool_before_attn1')(x)
    # MultiHeadAttention
    attn_out_1 = MultiHeadAttention(
        num_heads=4, 
        key_dim=64,  # 修改 key_dim 以匹配维度
        name='multi_head_attn_1'
    )(x, x)
    attn_out_1 = Dropout(0.2, name='dropout_attn1')(attn_out_1)
    # 残差连接
    x = Add(name='add_attn_res1')([x, attn_out_1])
    x = LayerNormalization(epsilon=1e-6, name='ln_attn1')(x)
    
    # 前馈网络(FFN)
    ffn_1 = Dense(256, activation='relu', name='ffn_1')(x)
    ffn_1 = Dense(256, name='ffn_2')(ffn_1)  # 保持维度一致
    x = Add(name='add_ffn_res1')([x, ffn_1])
    x = LayerNormalization(epsilon=1e-6, name='ln_ffn1')(x)

    #-----------------------------
    # 5. 第二个双向LSTM层
    #   进一步捕捉时序特征
    #-----------------------------
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        name='bilstm_2'
    )(x)
    x = BatchNormalization(name='bn_lstm2')(x)
    x = Dropout(0.2, name='dropout_lstm2')(x)

    #-----------------------------
    # 6. 第二次多头注意力
    #   捕捉更高层次的全局依赖
    #-----------------------------
    x = AveragePooling1D(pool_size=2, name='pool_before_attn2')(x)
    attn_out_2 = MultiHeadAttention(
        num_heads=4,
        key_dim=64,  # 修改 key_dim 以匹配维度
        name='multi_head_attn_2'
    )(x, x)
    attn_out_2 = Dropout(0.2, name='dropout_attn2')(attn_out_2)
    x = Add(name='add_attn_res2')([x, attn_out_2])
    x = LayerNormalization(epsilon=1e-6, name='ln_attn2')(x)
    
    # 前馈网络(FFN)
    ffn_2 = Dense(128, activation='relu', name='ffn_3')(x)
    ffn_2 = Dense(128, name='ffn_4')(ffn_2)  # 保持维度一致
    x = Add(name='add_ffn_res2')([x, ffn_2])
    x = LayerNormalization(epsilon=1e-6, name='ln_ffn2')(x)

    #-----------------------------
    # 7. 第三个双向LSTM层
    #   最终的特征提取
    #-----------------------------
    x = Bidirectional(
        LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        name='bilstm_3'
    )(x)
    x = BatchNormalization(name='bn_lstm3')(x)
    x = Dropout(0.2, name='dropout_lstm3')(x)

    #-----------------------------
    # 8. 双分支全局池化
    #   平均池化 + 最大池化
    #-----------------------------
    avg_pool = GlobalAveragePooling1D(name='avg_pool')(x)
    max_pool = GlobalMaxPooling1D(name='max_pool')(x)
    x = Concatenate(name='concat_pool')([avg_pool, max_pool])

    #-----------------------------
    # 9. 全连接层
    #   最终分类
    #-----------------------------
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = BatchNormalization(name='bn_dense1')(x)
    x = Dropout(0.3, name='dropout_dense1')(x)
    
    x = Dense(64, activation='relu', name='dense_2')(x)
    x = BatchNormalization(name='bn_dense2')(x)
    x = Dropout(0.3, name='dropout_dense2')(x)
    
    outputs = Dense(n_classes, activation=last_layer, name='output')(x)
    
    return Model(signal, outputs)

if __name__ == "__main__":
    # 测试模型
    model = get_model()
    model.summary()
