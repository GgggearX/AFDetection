import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def positional_encoding(position, d_model):
    """生成正余弦位置编码"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)
    # 偶数维度取 sin，奇数维度取 cos
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    """辅助函数，用于计算位置编码的角度"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

class ProbSparseSelfAttention(layers.Layer):
    """ProbSparse 自注意力（简化版）：使用多头注意力（全局注意力）作为近似实现"""
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(ProbSparseSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        # 线性层生成查询、键、值
        self.qkv = layers.Dense(embed_dim * 3, use_bias=False)
        # 输出线性层
        self.out = layers.Dense(embed_dim)

    def call(self, x):
        # 线性映射生成 Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, 3*embed_dim)
        q, k, v = tf.split(qkv, 3, axis=-1)  # 各 (batch, seq_len, embed_dim)
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        # 分离多头
        q = tf.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch, seq_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch, seq_len, self.num_heads, self.head_dim))
        # 转换为 (batch, num_heads, seq_len, head_dim)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        # 计算注意力得分
        attn_scores = tf.matmul(q, k, transpose_b=True)  # (batch, heads, seq_len, seq_len)
        attn_scores = attn_scores * self.scale
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        # 加权求和值
        attn_output = tf.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        # 合并多头输出
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # (batch, seq_len, heads, head_dim)
        attn_output = tf.reshape(attn_output, (batch, seq_len, self.embed_dim))
        # 线性输出
        output = self.out(attn_output)
        return output
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

class InformerBlock(layers.Layer):
    """Informer 编码块：包括自注意力和前馈网络"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(InformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention = ProbSparseSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # 多头自注意力
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

def get_model(max_seq_length, n_classes=1, last_layer='sigmoid', n_leads=1,
              num_heads=4, num_layers=3, d_model=64, dff=128, dropout_rate=0.1):
    """
    构建适用于 ECG 的 Informer 模型
    参数:
        max_seq_length: 输入序列长度
        n_classes: 输出类别数（1 表示二分类概率输出）
        last_layer: 最后一层激活函数
        n_leads: 导联数
        num_heads: 注意力头数
        num_layers: 编码块数量
        d_model: 嵌入维度
        dff: 前馈网络隐藏层维度
        dropout_rate: Dropout 比例
    """
    inputs = layers.Input(shape=(max_seq_length, n_leads))
    # 局部卷积特征提取
    x = layers.Conv1D(filters=d_model, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(filters=d_model, kernel_size=5, padding='same', activation='relu')(x)
    # 位置编码
    pos_encoding = positional_encoding(max_seq_length, d_model)
    x = layers.Dense(d_model)(x)  # 映射到 d_model 维度
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x = x + pos_encoding[:, :max_seq_length, :]
    x = layers.Dropout(dropout_rate)(x)
    # 堆叠 Informer 编码块
    for _ in range(num_layers):
        x = InformerBlock(d_model, num_heads, dff, dropout_rate)(x)
    # 全局平均池化 & 输出层
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation=last_layer)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # 测试模型结构
    model = get_model(max_seq_length=4096, n_classes=1, last_layer='sigmoid', n_leads=1)
    model.summary()
