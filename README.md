# 房颤检测系统 (AF Detection System)

这是一个基于深度学习的房颤（Atrial Fibrillation, AF）检测系统，使用集成学习方法结合RNN、DenseNet和Transformer模型来提高检测准确性。

## 项目结构

```
.
├── data/                   # 数据目录
│   ├── ecg_tracings.hdf5  # ECG数据
│   └── annotations/       # 标注数据
├── RNN/                   # RNN模型相关文件
│   ├── models/           # 保存的RNN模型
│   ├── results/          # 训练结果
│   ├── logs/             # 训练日志
│   └── preprocessed/     # 预处理数据
├── DenseNet/             # DenseNet模型相关文件
│   ├── models/          # 保存的DenseNet模型
│   ├── results/         # 训练结果
│   ├── logs/            # 训练日志
│   └── preprocessed/    # 预处理数据
├── Transformer/         # Transformer模型相关文件
│   ├── models/         # 保存的Transformer模型
│   ├── results/        # 训练结果
│   ├── logs/           # 训练日志
│   └── preprocessed/   # 预处理数据
├── models/              # 模型定义
│   ├── rnn_model.py    # RNN模型架构
│   ├── densenet_model.py # DenseNet模型架构
│   └── transformer_model.py # Transformer模型架构
├── train_rnn.py        # RNN模型训练脚本
├── train_densenet.py   # DenseNet模型训练脚本
├── train_transformer.py # Transformer模型训练脚本
├── ensemble_predict.py # 集成预测脚本
├── datasets.py         # 数据加载和预处理
└── visualize_results.py # 结果可视化
```

## 模型架构

### RNN模型
- 输入：12导联ECG信号 (4096个时间步)
- 三层双向LSTM：
  - 第一层：128个单元
  - 第二层：256个单元
  - 第三层：256个单元
- 每层LSTM后接：
  - 批归一化
  - Dropout(0.2)
- 全局平均池化
- 全连接层(256个单元)
- 输出层：sigmoid激活函数

### DenseNet模型
- 输入：12导联ECG信号 (4096个时间步)
- 多个Dense块，每个块包含：
  - 多个卷积层
  - 批归一化
  - ReLU激活
- 过渡层：
  - 1x1卷积
  - 2x2平均池化
- 全局平均池化
- 全连接层
- 输出层：sigmoid激活函数

### Transformer模型
- 输入：12导联ECG信号 (4096个时间步)
- 位置编码层
- 4个Transformer编码器块，每个块包含：
  - 多头自注意力机制（8个头，64维）
  - 前馈神经网络（256维）
  - 层归一化
  - Dropout(0.1)
- 全局平均池化
- 全连接层(256个单元)
- Dropout(0.5)
- 输出层：sigmoid激活函数

## 环境要求

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## 安装

1. 克隆仓库：
```bash
git clone git@github.com:GgggearX/AFDetection.git
cd AFDetection
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

1. 训练RNN模型：
```bash
python train_rnn.py data/ecg_tracings.hdf5 data/annotations/gold_standard.csv \
    --n_splits 5 \
    --epochs 70 \
    --batch_size 32 \
    --learning_rate 0.001
```
ecg_tracings.hdf5文件需自行下载后放入正确位置data/ecg_tracings.hdf5
2. 训练DenseNet模型：
```bash
python train_densenet.py data/ecg_tracings.hdf5 data/annotations/gold_standard.csv \
    --n_splits 5 \
    --epochs 70 \
    --batch_size 32 \
    --learning_rate 0.001
```

3. 训练Transformer模型：
```bash
python train_transformer.py data/ecg_tracings.hdf5 data/annotations/gold_standard.csv \
    --n_splits 5 \
    --epochs 70 \
    --batch_size 32 \
    --learning_rate 0.001
```

### 预测

使用集成模型进行预测：
```bash
python ensemble_predict.py data/test_data.hdf5
```

## 模型性能

- 使用5折交叉验证
- 早停策略：验证损失10轮无改善则停止
- 学习率调整：验证损失5轮无改善则降低为原来的0.1倍
- 使用AUC和准确率作为评估指标
- 集成预测：三个模型的预测结果取平均值

## 注意事项

1. 确保有足够的GPU内存（推荐至少8GB）
2. 训练过程中会自动保存最佳模型
3. 可以使用TensorBoard查看训练过程：
```bash
tensorboard --logdir=RNN/logs
```
4. ecg_tracings.hdf5文件需自行下载后放入正确位置data/ecg_tracings.hdf5
