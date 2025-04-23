import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    生成位置编码
    
    Args:
        position: 序列长度
        d_model: 模型维度
    
    Returns:
        位置编码矩阵
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    
    # 将正弦应用于偶数索引
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # 将余弦应用于奇数索引
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder_layer(inputs, d_model, num_heads, dff, dropout_rate=0.1):
    """
    Transformer编码器层
    
    Args:
        inputs: 输入张量
        d_model: 模型维度
        num_heads: 注意力头数
        dff: 前馈网络维度
        dropout_rate: Dropout率
    
    Returns:
        编码器层输出
    """
    # 多头自注意力
    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model
    )(inputs, inputs)
    attention = layers.Dropout(dropout_rate)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # 前馈网络
    ffn = layers.Dense(dff, activation='relu')(attention)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    
    return layers.LayerNormalization(epsilon=1e-6)(attention + ffn)

def get_model(max_seq_length, n_classes=1, last_layer='sigmoid', n_leads=1, num_heads=8, num_layers=4, d_model=64, dff=256, dropout_rate=0.1):
    """
    构建Transformer模型
    
    Args:
        max_seq_length: 最大序列长度
        n_classes: 输出类别数
        last_layer: 最后一层激活函数
        n_leads: 导联数
        num_heads: 注意力头数
        num_layers: Transformer层数
        d_model: 模型维度
        dff: 前馈网络维度
        dropout_rate: Dropout率
    
    Returns:
        Transformer模型
    """
    inputs = layers.Input(shape=(max_seq_length, n_leads))
    
    # 位置编码
    pos_encoding = positional_encoding(max_seq_length, d_model)
    x = layers.Dense(d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += pos_encoding[:, :max_seq_length, :]
    
    # Transformer层
    for _ in range(num_layers):
        x = transformer_encoder_layer(x, d_model, num_heads, dff, dropout_rate)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling1D()(x)
    
    # 输出层
    outputs = layers.Dense(n_classes, activation=last_layer)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,
        clipvalue=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc_1')]
    )
    
    return model 