# 心房颤动检测系统

基于深度学习的心房颤动（AF）自动检测系统，包含多种深度学习模型，包括RNN、DenseNet、CNN-LSTM、WaveNet和Transformer模型。

## 项目结构

```
.
├── data/
│   └── physionet/
│       └── training2017/         # 训练数据集
│           ├── *.mat            # ECG信号文件
│           └── REFERENCE.csv    # 标签文件
├── models/                      # 模型定义
│   ├── rnn_model.py
│   ├── densenet_model.py
│   ├── cnn_lstm_model.py
│   ├── wavenet_model.py
│   └── transformer_model.py
├── outputs/                     # 模型输出
│   ├── RNN/
│   ├── DenseNet/
│   ├── CNN-LSTM/
│   ├── WaveNet/
│   └── Transformer/
├── datasets.py                  # 数据加载和预处理
├── train_rnn.py                # RNN模型训练脚本
├── train_densenet.py           # DenseNet模型训练脚本
├── train_cnn_lstm.py           # CNN-LSTM模型训练脚本
├── train_wavenet.py            # WaveNet模型训练脚本
├── train_transformer.py        # Transformer模型训练脚本
├── visualize_results.py        # 结果可视化脚本
└── requirements.txt            # 项目依赖
```

## 模型说明

### 1. RNN模型
- 使用双向LSTM层进行序列特征提取
- 添加注意力机制增强关键特征的学习
- 使用全连接层进行最终分类

### 2. DenseNet模型
- 采用密集连接的卷积块结构
- 实现特征的高效重用和梯度的良好传播
- 使用全局平均池化减少参数量

### 3. CNN-LSTM模型
- 结合CNN的特征提取能力和LSTM的时序建模能力
- 使用残差连接改善梯度流动
- 添加Dropout层防止过拟合

### 4. WaveNet模型
- 使用扩张卷积实现大感受野
- 采用残差学习和跳跃连接
- 通过因果卷积保持时序特性

### 5. Transformer模型
- 使用多头自注意力机制捕获全局依赖关系
- 采用位置编码保留序列位置信息
- 实现并行计算提高训练效率

## 训练流程

每个模型的训练过程都采用5折交叉验证，并包含以下步骤：

1. 数据预处理：
   - 信号归一化
   - 固定长度裁剪/填充
   - 数据增强（可选）
   - SMOTE处理类别不平衡

2. 模型训练：
   ```bash
   python train_rnn.py
   python train_densenet.py
   python train_cnn_lstm.py
   python train_wavenet.py
   python train_transformer.py
   ```

3. 训练过程监控：
   - 实时显示训练进度
   - 记录训练和验证指标
   - 自动保存最佳模型

4. 模型评估：
   - ROC曲线分析
   - 混淆矩阵
   - 准确率、精确率、召回率等指标

## 结果可视化

使用 `visualize_results.py` 生成可视化结果：

```bash
python visualize_results.py --models RNN DenseNet CNN-LSTM WaveNet Transformer --output_dir model_comparison
```

可视化内容：
1. 训练历史曲线
   - 损失函数变化
   - 准确率变化
   - AUC变化

2. 模型性能对比
   - 各模型F1得分对比
   - 各模型AUC、精确率、召回率对比
   - 性能指标雷达图

3. 预测结果分析
   - 混淆矩阵热力图
   - 指标相关性热力图
   - 学习曲线可视化

## 输出目录结构

每个模型的输出目录包含：

```
model_name/
├── models/                 # 模型权重
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
├── logs/                  # 训练日志
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
└── results/              # 评估结果
    ├── metrics_comparison.png
    ├── learning_curves_grid.png
    ├── model_radar_chart.png
    ├── metrics_correlation.png
    └── val_f1_score_learning_curves.png
```

## 环境要求

- Python 3.7+
- TensorFlow 2.6+
- CUDA 11.0+（推荐使用GPU加速）
- 其他依赖见 requirements.txt

## 安装说明

1. 创建虚拟环境：
```bash
conda create -n afdetection python=3.7.16
conda activate afdetection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用建议

1. 数据准备
   - 确保数据格式正确
   - 检查标签分布是否均衡
   - 考虑使用SMOTE处理类别不平衡

2. 训练过程
   - 根据GPU内存调整batch size
   - 适当调整学习率和训练轮数
   - 使用Focal Loss处理类别不平衡

3. 模型选择
   - 可以单独使用某个模型
   - 根据实际需求选择合适的模型
   - 考虑推理速度和准确率的权衡

4. 结果分析
   - 关注验证集性能
   - 分析错误预测案例
   - 考虑模型的计算开销

## 注意事项

1. 内存使用
   - 大型模型训练需要足够的GPU内存
   - 数据生成器支持批量加载

2. 训练时间
   - Transformer和WaveNet训练较慢
   - 可以适当减少训练轮数

3. 模型保存
   - 定期保存检查点
   - 保存最佳模型权重

## 常见问题

1. 内存不足
   - 减小batch size
   - 使用数据生成器
   - 选择较小的模型

2. 训练不稳定
   - 调整学习率
   - 使用梯度裁剪
   - 检查数据预处理

3. 预测效果不佳
   - 增加训练数据
   - 调整模型架构
   - 尝试不同的数据增强方法

