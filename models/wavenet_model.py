import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class CausalConv1D(layers.Layer):
    """因果卷积层，确保输出只依赖于当前及之前的输入"""
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(CausalConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate

    def build(self, input_shape):
        self.conv = layers.Conv1D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   dilation_rate=self.dilation_rate,
                                   padding='valid', use_bias=True)
        super(CausalConv1D, self).build(input_shape)

    def call(self, inputs):
        # 左侧填充，以保持因果性
        padded = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])
        return self.conv(padded)

    def get_config(self):
        config = super(CausalConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config

class WaveNetBlock(layers.Layer):
    """WaveNet残差块"""
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super(WaveNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
        self.causal_conv = CausalConv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv1D(filters=self.filters, kernel_size=1, kernel_initializer=kernel_initializer)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv1D(filters=self.filters, kernel_size=1, kernel_initializer=kernel_initializer)
        self.bn3 = layers.BatchNormalization()
        # skip连接的1x1卷积
        self.skip_conv = layers.Conv1D(filters=self.filters, kernel_size=1, kernel_initializer=kernel_initializer)
        super(WaveNetBlock, self).build(input_shape)

    def call(self, inputs):
        # 主路径卷积运算
        x = self.causal_conv(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        # 残差连接
        res = layers.Add()([inputs, x])
        # Skip连接输出
        skip = self.skip_conv(x)
        return res, skip

    def get_config(self):
        config = super(WaveNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config

def get_model(max_seq_length, n_classes=1, last_layer='sigmoid', n_leads=1,
              num_blocks=4, num_filters=64, kernel_size=32):
    """
    构建WaveNet模型
    """
    kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
    inputs = layers.Input(shape=(max_seq_length, n_leads))
    # 初始1x1卷积层
    x = layers.Conv1D(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 残差块堆叠
    skip_connections = []
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x, skip = WaveNetBlock(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate)(x)
        if i < num_blocks - 1:
            x = layers.Dropout(0.2)(x)  # 每个残差块输出后添加Dropout
        skip_connections.append(skip)
    # 合并所有skip连接
    x = layers.Add()(skip_connections)
    # 后处理卷积和激活
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 全局池化 + 输出
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = get_model(max_seq_length=1024, n_classes=1, last_layer='sigmoid', n_leads=1)
    model.summary()

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class CausalConv1D(layers.Layer):
    """因果卷积层，确保输出只依赖于当前及之前的输入"""
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(CausalConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate

    def build(self, input_shape):
        self.conv = layers.Conv1D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   dilation_rate=self.dilation_rate,
                                   padding='valid', use_bias=True)
        super(CausalConv1D, self).build(input_shape)

    def call(self, inputs):
        # 左侧填充，以保持因果性
        padded = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])
        return self.conv(padded)

    def get_config(self):
        config = super(CausalConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config

class WaveNetBlock(layers.Layer):
    """WaveNet残差块"""
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super(WaveNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
        self.causal_conv = CausalConv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv1D(filters=self.filters, kernel_size=1, kernel_initializer=kernel_initializer)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv1D(filters=self.filters, kernel_size=1, kernel_initializer=kernel_initializer)
        self.bn3 = layers.BatchNormalization()
        # skip连接的1x1卷积
        self.skip_conv = layers.Conv1D(filters=self.filters, kernel_size=1, kernel_initializer=kernel_initializer)
        super(WaveNetBlock, self).build(input_shape)

    def call(self, inputs):
        # 主路径卷积运算
        x = self.causal_conv(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        # 残差连接
        res = layers.Add()([inputs, x])
        # Skip连接输出
        skip = self.skip_conv(x)
        return res, skip

    def get_config(self):
        config = super(WaveNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config

def get_model(max_seq_length, n_classes=1, last_layer='sigmoid', n_leads=1,
              num_blocks=4, num_filters=64, kernel_size=32):
    """
    构建WaveNet模型
    """
    kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
    inputs = layers.Input(shape=(max_seq_length, n_leads))
    # 初始1x1卷积层
    x = layers.Conv1D(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 残差块堆叠
    skip_connections = []
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x, skip = WaveNetBlock(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate)(x)
        if i < num_blocks - 1:
            x = layers.Dropout(0.2)(x)  # 每个残差块输出后添加Dropout
        skip_connections.append(skip)
    # 合并所有skip连接
    x = layers.Add()(skip_connections)
    # 后处理卷积和激活
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 全局池化 + 输出
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = get_model(max_seq_length=1024, n_classes=1, last_layer='sigmoid', n_leads=1)
    model.summary()
