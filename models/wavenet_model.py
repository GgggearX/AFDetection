import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class CausalConv1D(layers.Layer):
    """因果卷积层，确保输出只依赖于当前时刻及之前的输入"""
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(CausalConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate
        
    def build(self, input_shape):
        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='valid',
            use_bias=True
        )
        super(CausalConv1D, self).build(input_shape)
        
    def call(self, inputs):
        # 添加padding以保持序列长度
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
    """WaveNet的基本构建块"""
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super(WaveNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
    def build(self, input_shape):
        # 使用He初始化
        kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
        
        self.causal_conv = CausalConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=kernel_initializer
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=kernel_initializer
        )
        self.bn3 = layers.BatchNormalization()
        
        # 添加skip connection的卷积层
        self.skip_conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=kernel_initializer
        )
        
        super(WaveNetBlock, self).build(input_shape)
        
    def call(self, inputs):
        # 主路径
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
        
        # Skip connection
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

def build_wavenet_model(input_shape, num_classes=1, num_blocks=4, num_filters=32, kernel_size=3):
    """
    构建WaveNet模型
    
    Args:
        input_shape: 输入形状 (time_steps, channels)
        num_classes: 输出类别数
        num_blocks: WaveNet块的数量
        num_filters: 每个卷积层的滤波器数量
        kernel_size: 卷积核大小
    
    Returns:
        WaveNet模型
    """
    # 使用He初始化
    kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
    
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷积层
    x = layers.Conv1D(
        filters=num_filters, 
        kernel_size=1,
        kernel_initializer=kernel_initializer
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # WaveNet块
    skip_connections = []
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x, skip = WaveNetBlock(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )(x)
        skip_connections.append(skip)
    
    # 合并所有skip connections
    x = layers.Add()(skip_connections)
    
    # 最终处理
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(
        filters=num_filters, 
        kernel_size=1,
        kernel_initializer=kernel_initializer
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling1D()(x)
    
    # 输出层
    outputs = layers.Dense(
        num_classes, 
        activation='sigmoid',
        kernel_initializer=kernel_initializer
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_wavenet_model(input_shape, learning_rate=0.001):
    """
    创建并编译WaveNet模型
    
    Args:
        input_shape: 输入形状
        learning_rate: 学习率
    
    Returns:
        编译好的WaveNet模型
    """
    model = build_wavenet_model(input_shape)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,  # 添加梯度裁剪
        clipvalue=0.5  # 限制梯度值范围
    )
    
    # 使用自定义的AUC指标名称
    auc_metric = tf.keras.metrics.AUC(name='auc_1')
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', auc_metric]
    )
    
    return model 