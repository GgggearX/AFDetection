#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG原始数据集分布分析工具
直接分析未经预处理的原始ECG数据，包括信号特征和可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import argparse
from tqdm import tqdm
import sys
import h5py
import traceback
import json

# 尝试导入wfdb库，用于读取PhysioNet格式数据
try:
    import wfdb
    from scipy.signal import resample
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("警告: wfdb库未安装，将无法读取PhysioNet格式数据。请使用 'pip install wfdb' 安装。")

# 设置绘图风格
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_output_dir(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    subdirs = ['class_distribution', 'signal_features', 'time_domain', 'frequency_domain', 'raw_data']
    for subdir in subdirs:
        path = os.path.join(output_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    return output_dir

def load_data_from_hdf5(data_path, labels_path=None):
    """
    直接从HDF5文件加载原始数据和标签，不进行任何预处理
    """
    print(f"从HDF5文件加载原始数据: {data_path}")
    
    try:
        # 加载ECG数据
        with h5py.File(data_path, 'r') as f:
            # 尝试找到数据
            if 'tracings' in f:
                ecg_data = np.array(f['tracings'])
                print(f"从'tracings'加载数据, 形状: {ecg_data.shape}")
            else:
                # 列出所有键并尝试第一个
                keys = list(f.keys())
                if keys:
                    ecg_data = np.array(f[keys[0]])
                    print(f"从'{keys[0]}'加载数据, 形状: {ecg_data.shape}")
                else:
                    raise ValueError(f"在{data_path}中未找到数据")
        
        # 加载标签
        if labels_path and os.path.exists(labels_path):
            try:
                # 尝试CSV格式标签文件
                labels_df = pd.read_csv(labels_path)
                
                # 检查是否是多标签格式
                if 'AF' in labels_df.columns:
                    print("找到AF标签列")
                    labels = labels_df['AF'].values.astype(np.float32)
                # 检查是否为金标准格式
                elif 'label' in labels_df.columns:
                    print("找到label标签列")
                    labels = (labels_df['label'] == 'A').astype(np.float32)
                # 尝试从第二列获取标签
                elif labels_df.shape[1] >= 2:
                    print("从第二列获取标签")
                    labels = (labels_df.iloc[:, 1] == 'A').astype(np.float32)
                else:
                    raise ValueError("无法从标签文件中确定标签")
                
                print(f"加载标签成功, 形状: {labels.shape}, 阳性率: {np.mean(labels):.4f}")
                
                # 确保数据和标签数量匹配
                if len(labels) != len(ecg_data):
                    print(f"数据({len(ecg_data)})和标签({len(labels)})数量不匹配，将截断到较短的长度")
                    min_len = min(len(ecg_data), len(labels))
                    ecg_data = ecg_data[:min_len]
                    labels = labels[:min_len]
                
                return ecg_data, labels, None, None
            
            except Exception as e:
                print(f"加载标签文件失败: {str(e)}")
                print(traceback.format_exc())
                # 如果标签加载失败，仍然返回数据，但标签为None
                return ecg_data, None, None, None
        else:
            print(f"未提供标签文件或标签文件不存在")
            return ecg_data, None, None, None
            
    except Exception as e:
        print(f"从HDF5文件加载数据失败: {str(e)}")
        print(traceback.format_exc())
        raise

def load_from_wfdb(data_dir, labels_file=None, max_seq_length=None):
    """
    加载PhysioNet WFDB格式的原始数据和标签，保持原始采样率
    
    参数:
        data_dir: 数据目录路径
        labels_file: 标签文件路径
        max_seq_length: 如果设置，则截断或填充到此长度，否则保持原始长度
    
    返回:
        (x_data, y_data, sample_rates, record_lengths) 元组
    """
    print(f"从WFDB格式加载原始数据: {data_dir}")
    
    if not WFDB_AVAILABLE:
        raise ImportError("无法加载WFDB格式数据，请安装wfdb库: pip install wfdb")
    
    try:
        # 加载标签文件
        if labels_file and os.path.exists(labels_file):
            print(f"从 {labels_file} 加载标签文件...")
            ref_df = pd.read_csv(labels_file, header=None)
            
            # 根据列数判断格式
            if ref_df.shape[1] >= 2:
                # 添加标准列名
                if ref_df.shape[1] == 2:
                    ref_df.columns = ['filename', 'label']
                else:
                    ref_df = ref_df.iloc[:, 0:2]
                    ref_df.columns = ['filename', 'label']
                
                # 将标签转换为二进制(房颤为1，其他为0)
                ref_df['af_label'] = ref_df['label'].apply(lambda x: 1 if x == 'A' else 0)
                print(f"标签文件加载完成，共 {len(ref_df)} 条记录")
            else:
                raise ValueError(f"标签文件格式不正确: {labels_file}")
        else:
            print(f"未提供标签文件或标签文件不存在: {labels_file}")
            # 搜索目录下的所有记录
            record_files = [f for f in os.listdir(data_dir) if f.endswith('.hea')]
            filenames = [os.path.splitext(f)[0] for f in record_files]
            ref_df = pd.DataFrame({'filename': filenames})
            ref_df['af_label'] = np.nan  # 未知标签
            print(f"未找到标签文件，将处理目录中的 {len(ref_df)} 个记录，但不包含标签信息")
        
        # 加载数据
        data_list = []
        labels = []
        sample_rates = []
        record_lengths = []
        count = 0
        
        for i, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="加载WFDB数据"):
            if count % 100 == 0:
                print(f"已处理 {count}/{len(ref_df)} 条记录")
            
            count += 1
            filename = row['filename']
            
            try:
                # 读取数据
                file_path = os.path.join(data_dir, filename)
                record = wfdb.rdrecord(file_path)
                
                # 获取信号并确保是二维数组
                signal_data = record.p_signal
                
                # 记录原始形状和重新形状(如果需要)
                if len(signal_data.shape) == 1:  # 单导联一维数据
                    # 转换为二维(序列长度, 1)
                    signals = signal_data.reshape(-1, 1)
                    seq_length = len(signal_data)
                    n_leads = 1
                else:  # 已经是二维数组(样本点, 导联数)
                    signals = signal_data
                    seq_length = signals.shape[0]
                    n_leads = signals.shape[1]
                
                # 对于数据分析，我们保持(序列长度, 导联数)的格式
                # 注意：这与之前的(导联数, 序列长度)格式不同
                
                # 保存原始记录信息
                sample_rates.append(record.fs)
                record_lengths.append(seq_length)
                
                # 如果指定了最大序列长度，进行相应处理
                if max_seq_length is not None:
                    if seq_length > max_seq_length:
                        # 截断过长的信号
                        signals = signals[:max_seq_length, :]
                    elif seq_length < max_seq_length:
                        # 填充过短的信号
                        pad_width = max_seq_length - seq_length
                        signals = np.pad(signals, ((0, pad_width), (0, 0)), 'constant')
                
                # 添加到数据列表
                data_list.append(signals)
                
                # 如果有标签，记录标签
                if 'af_label' in row and not pd.isna(row['af_label']):
                    labels.append(int(row['af_label']))
                
            except Exception as e:
                print(f"加载文件 {filename} 时出错: {e}")
                continue
        
        # 将列表转换为NumPy数组，使用dtype=object处理不同形状的序列
        print(f"数据列表长度: {len(data_list)}")
        
        if len(data_list) > 0:
            # 检查第一个元素的形状
            first_shape = data_list[0].shape
            print(f"第一个数据项形状: {first_shape}")
            
            # 检查是否所有元素都具有相同的形状
            same_shape = all(item.shape == first_shape for item in data_list)
            if same_shape:
                print(f"所有数据具有相同形状: {first_shape}，使用常规数组")
                x_data = np.array(data_list)
            else:
                print("数据具有不同形状，使用对象数组")
                x_data = np.array(data_list, dtype=object)
        else:
            print("警告: 数据列表为空")
            x_data = np.array([], dtype=object)
        
        print(f"数据加载完成, 数据项数量: {len(x_data)}")
        if sample_rates:
            print(f"采样率统计: 最小={min(sample_rates)}, 最大={max(sample_rates)}, 平均={np.mean(sample_rates):.1f}")
        if record_lengths:
            print(f"记录长度统计: 最小={min(record_lengths)}, 最大={max(record_lengths)}, 平均={np.mean(record_lengths):.1f}")
        
        # 如果有标签，返回标签
        if labels and len(labels) == len(x_data):
            y_data = np.array(labels)
            print(f"标签加载完成, 形状: {y_data.shape}, 阳性率: {np.mean(y_data):.4f}")
            return x_data, y_data, sample_rates, record_lengths
        else:
            print("没有匹配的标签或标签数量不匹配")
            return x_data, None, sample_rates, record_lengths
        
    except Exception as e:
        print(f"从WFDB格式加载数据失败: {str(e)}")
        print(traceback.format_exc())
        raise

def load_raw_data(data_path, labels_path=None, max_seq_length=None):
    """
    加载原始数据和标签，自动检测数据类型
    """
    print(f"加载原始数据: {data_path}")
    
    # 验证路径存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    # 根据文件类型/目录加载数据
    if os.path.isdir(data_path):
        # 如果是目录，尝试作为WFDB目录加载
        print(f"检测到目录路径，尝试作为WFDB数据目录加载: {data_path}")
        return load_from_wfdb(data_path, labels_path, max_seq_length)
    elif data_path.endswith('.hdf5') or data_path.endswith('.h5'):
        return load_data_from_hdf5(data_path, labels_path)
    elif data_path.endswith('.npy'):
        try:
            # 加载numpy数组
            x_data = np.load(data_path)
            print(f"从NPY文件加载数据成功, 形状: {x_data.shape}")
            
            # 尝试加载标签
            if labels_path and os.path.exists(labels_path):
                if labels_path.endswith('.npy'):
                    y_true = np.load(labels_path)
                    print(f"从NPY文件加载标签成功, 形状: {y_true.shape}")
                else:
                    # 尝试CSV格式
                    labels_df = pd.read_csv(labels_path)
                    if 'AF' in labels_df.columns:
                        y_true = labels_df['AF'].values
                    elif 'label' in labels_df.columns:
                        y_true = (labels_df['label'] == 'A').astype(int).values
                    else:
                        y_true = None
                        print(f"无法从CSV标签文件中确定标签列: {labels_path}")
                
                if y_true is not None:
                    print(f"标签加载成功, 形状: {y_true.shape}, 阳性率: {np.mean(y_true):.4f}")
            else:
                y_true = None
                print("未提供标签文件或标签文件不存在")
            
            # 对于NPY文件，我们没有采样率和记录长度信息
            return x_data, y_true, None, None
        
        except Exception as e:
            print(f"从NPY文件加载数据失败: {str(e)}")
            print(traceback.format_exc())
            raise
    else:
        raise ValueError(f"不支持的数据文件格式: {data_path}, 支持的格式: .hdf5, .h5, .npy 或 WFDB目录")

def analyze_class_distribution(y_data, output_dir):
    """分析并可视化类别分布"""
    print("分析类别分布...")
    class_counts = np.bincount(y_data.astype(int))
    class_names = ['非房颤', '房颤']
    class_percentages = class_counts / len(y_data) * 100
    
    # 绘制饼图
    plt.figure(figsize=(10, 6))
    plt.pie(class_counts, labels=class_names, autopct='%1.1f%%', 
            colors=['#4CAF50', '#F44336'], startangle=90, explode=(0, 0.1),
            shadow=True, textprops={'fontsize': 14})
    plt.title('数据集类别分布', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution', 'class_pie_chart.png'), dpi=300)
    plt.close()
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_counts, color=['#4CAF50', '#F44336'])
    plt.title('数据集类别计数', fontsize=16)
    plt.ylabel('样本数量', fontsize=14)
    
    # 为条形图添加标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}\n({height/sum(class_counts):.1%})',
                ha='center', va='bottom', fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution', 'class_bar_chart.png'), dpi=300)
    plt.close()
    
    # 保存分布摘要
    distribution_summary = pd.DataFrame({
        '类别': class_names,
        '样本数': class_counts,
        '百分比': [f'{p:.2f}%' for p in class_percentages]
    })
    distribution_summary.to_csv(os.path.join(output_dir, 'class_distribution', 'class_distribution.csv'), 
                               index=False, encoding='utf-8')
    
    print(f"类别分布: 房颤 {class_counts[1]} ({class_percentages[1]:.2f}%), "
          f"非房颤 {class_counts[0]} ({class_percentages[0]:.2f}%)")
    return class_counts

def analyze_signal_features(x_data, y_data, output_dir):
    """分析信号特征分布"""
    print("分析信号特征...")
    
    # 检查是否为对象数组
    is_object_array = isinstance(x_data, np.ndarray) and x_data.dtype == object
    
    # 提取信号特征
    n_samples = len(x_data)
    
    if is_object_array:
        print("检测到对象数组，将单独处理每个样本...")
        # 对于对象数组，需要单独处理每个样本
        signal_means = []
        signal_stds = []
        signal_mins = []
        signal_maxs = []
        
        for i in range(n_samples):
            sample = x_data[i]
            if len(sample.shape) == 2:  # (序列长度, 导联数) 或 (导联数, 序列长度)
                # 假设格式为 (序列长度, 导联数)
                means = np.mean(sample, axis=0)
                stds = np.std(sample, axis=0)
                mins = np.min(sample, axis=0)
                maxs = np.max(sample, axis=0)
            else:  # 一维数组
                means = np.mean(sample)
                stds = np.std(sample)
                mins = np.min(sample)
                maxs = np.max(sample)
            
            signal_means.append(means)
            signal_stds.append(stds)
            signal_mins.append(mins)
            signal_maxs.append(maxs)
        
        # 将结果转换为NumPy数组
        signal_means = np.array(signal_means)
        signal_stds = np.array(signal_stds)
        signal_mins = np.array(signal_mins)
        signal_maxs = np.array(signal_maxs)
    else:
        # 原始逻辑用于常规NumPy数组
        seq_length = x_data.shape[1]
        n_leads = x_data.shape[2] if len(x_data.shape) > 2 else 1
        
        # 计算基本统计量
        signal_means = np.mean(x_data, axis=1)
        signal_stds = np.std(x_data, axis=1)
        signal_mins = np.min(x_data, axis=1)
        signal_maxs = np.max(x_data, axis=1)
    
    # 计算信号范围
    signal_ranges = signal_maxs - signal_mins
    
    # 根据类别划分特征
    af_indices = np.where(y_data == 1)[0]
    non_af_indices = np.where(y_data == 0)[0]
    
    # 绘制信号均值分布直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(signal_means.flatten(), bins=50, kde=True, color='#3498db', label='所有样本')
    if len(af_indices) > 0:
        sns.histplot(signal_means[af_indices].flatten(), bins=50, kde=True, color='#e74c3c', label='房颤')
    if len(non_af_indices) > 0:
        sns.histplot(signal_means[non_af_indices].flatten(), bins=50, kde=True, color='#2ecc71', label='非房颤')
    plt.title('信号均值分布', fontsize=16)
    plt.xlabel('信号均值', fontsize=14)
    plt.ylabel('频数', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_features', 'signal_mean_distribution.png'), dpi=300)
    plt.close()
    
    # 绘制信号标准差分布直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(signal_stds.flatten(), bins=50, kde=True, color='#3498db', label='所有样本')
    if len(af_indices) > 0:
        sns.histplot(signal_stds[af_indices].flatten(), bins=50, kde=True, color='#e74c3c', label='房颤')
    if len(non_af_indices) > 0:
        sns.histplot(signal_stds[non_af_indices].flatten(), bins=50, kde=True, color='#2ecc71', label='非房颤')
    plt.title('信号标准差分布', fontsize=16)
    plt.xlabel('信号标准差', fontsize=14)
    plt.ylabel('频数', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_features', 'signal_std_distribution.png'), dpi=300)
    plt.close()
    
    # 绘制信号范围分布直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(signal_ranges.flatten(), bins=50, kde=True, color='#3498db', label='所有样本')
    if len(af_indices) > 0:
        sns.histplot(signal_ranges[af_indices].flatten(), bins=50, kde=True, color='#e74c3c', label='房颤')
    if len(non_af_indices) > 0:
        sns.histplot(signal_ranges[non_af_indices].flatten(), bins=50, kde=True, color='#2ecc71', label='非房颤')
    plt.title('信号幅值范围分布', fontsize=16)
    plt.xlabel('信号幅值范围', fontsize=14)
    plt.ylabel('频数', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_features', 'signal_range_distribution.png'), dpi=300)
    plt.close()
    
    # 绘制箱线图比较房颤和非房颤信号
    features = {
        '信号均值': signal_means.flatten(),
        '信号标准差': signal_stds.flatten(),
        '信号幅值范围': signal_ranges.flatten()
    }
    
    for feature_name, feature_values in features.items():
        plt.figure(figsize=(10, 6))
        data = {
            '类别': ['非房颤'] * len(non_af_indices) + ['房颤'] * len(af_indices),
            feature_name: np.concatenate([feature_values[non_af_indices], feature_values[af_indices]])
        }
        df = pd.DataFrame(data)
        sns.boxplot(x='类别', y=feature_name, data=df, palette={'非房颤': '#2ecc71', '房颤': '#e74c3c'})
        plt.title(f'房颤与非房颤信号的{feature_name}对比', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'signal_features', f'{feature_name}_boxplot.png'), dpi=300)
        plt.close()
    
    if is_object_array:
        print(f"信号特征分析完成: 样本数量={n_samples}, 数据类型=对象数组")
    else:
        seq_length = x_data.shape[1]
        n_leads = x_data.shape[2] if len(x_data.shape) > 2 else 1
        print(f"信号特征分析完成: 样本数量={n_samples}, 序列长度={seq_length}, 导联数={n_leads}")

def visualize_time_domain(x_data, y_data, output_dir, num_samples=5):
    """可视化时域信号"""
    print("可视化时域信号...")
    
    # 如果没有标签，创建虚拟标签进行处理
    if y_data is None:
        y_data = np.zeros(len(x_data))
    
    # 选择几个房颤和非房颤样本
    af_indices = np.where(y_data == 1)[0]
    non_af_indices = np.where(y_data == 0)[0]
    
    # 随机选择样本
    np.random.seed(42)  # 设置随机种子以确保可重复性
    selected_af = np.random.choice(af_indices, min(num_samples, len(af_indices)), replace=False) if len(af_indices) > 0 else []
    selected_non_af = np.random.choice(non_af_indices, min(num_samples, len(non_af_indices)), replace=False) if len(non_af_indices) > 0 else []
    
    # 检查数据类型和形状
    is_object_array = isinstance(x_data, np.ndarray) and x_data.dtype == object
    
    if is_object_array:
        # 对于object数组，取第一个样本确定结构
        if len(x_data) > 0:
            first_sample = x_data[0]
            if len(first_sample.shape) == 2:
                n_leads = min(first_sample.shape)  # 假设导联数是较小的维度
                seq_length = max(first_sample.shape)  # 序列长度是较大的维度
                is_leads_first = first_sample.shape[0] < first_sample.shape[1]  # 导联是否是第一维
            else:
                n_leads = 1
                seq_length = first_sample.shape[0]
                is_leads_first = False
        else:
            print("数据集为空，无法可视化")
            return
    else:
        # 获取信号长度和导联数
        if x_data.ndim == 3:
            if x_data.shape[1] <= 12:  # 假设为 (样本数, 导联数, 序列长度)
                n_leads = x_data.shape[1]
                seq_length = x_data.shape[2]
                is_leads_first = True
            else:  # 假设为 (样本数, 序列长度, 导联数)
                n_leads = x_data.shape[2]
                seq_length = x_data.shape[1]
                is_leads_first = False
        else:  # 单导联 (样本数, 序列长度)
            n_leads = 1
            seq_length = x_data.shape[1]
            is_leads_first = False
    
    # 为信号创建时间轴
    time_axis = np.arange(seq_length)
    
    # 创建不同颜色的循环
    lead_colors = plt.cm.tab10(np.linspace(0, 1, n_leads))
    
    # 可视化房颤信号
    for i, idx in enumerate(selected_af):
        plt.figure(figsize=(15, 4))
        
        if is_object_array:
            sample = x_data[idx]
            # 对于异构数据，可能需要创建特定长度的时间轴
            if len(sample.shape) == 2:
                if is_leads_first:  # (n_leads, seq_length)
                    sample_seq_length = sample.shape[1]
                    sample_n_leads = sample.shape[0]
                else:  # (seq_length, n_leads)
                    sample_seq_length = sample.shape[0]
                    sample_n_leads = sample.shape[1]
                sample_time_axis = np.arange(sample_seq_length)
                
                if is_leads_first:
                    for lead in range(min(n_leads, sample_n_leads)):
                        plt.plot(sample_time_axis, sample[lead], label=f'导联 {lead+1}', color=lead_colors[lead])
                else:
                    for lead in range(min(n_leads, sample_n_leads)):
                        plt.plot(sample_time_axis, sample[:, lead], label=f'导联 {lead+1}', color=lead_colors[lead])
            else:  # 单导联
                sample_time_axis = np.arange(len(sample))
                plt.plot(sample_time_axis, sample, label='导联 1', color=lead_colors[0])
        else:
            sample = x_data[idx]
            if len(sample.shape) == 1:  # 单导联
                plt.plot(time_axis, sample, label='导联 1', color=lead_colors[0])
            else:  # 多导联
                for lead in range(n_leads):
                    if is_leads_first:  # (n_leads, seq_length)
                        plt.plot(time_axis, sample[lead], label=f'导联 {lead+1}', color=lead_colors[lead])
                    else:  # (seq_length, n_leads)
                        plt.plot(time_axis, sample[:, lead], label=f'导联 {lead+1}', color=lead_colors[lead])
        
        plt.title(f'房颤信号示例 #{i+1}', fontsize=16)
        plt.xlabel('样本点', fontsize=14)
        plt.ylabel('幅值', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_domain', f'af_signal_example_{i+1}.png'), dpi=300)
        plt.close()
    
    # 可视化非房颤信号
    for i, idx in enumerate(selected_non_af):
        plt.figure(figsize=(15, 4))
        
        if is_object_array:
            sample = x_data[idx]
            # 对于异构数据，可能需要创建特定长度的时间轴
            if len(sample.shape) == 2:
                if is_leads_first:  # (n_leads, seq_length)
                    sample_seq_length = sample.shape[1]
                    sample_n_leads = sample.shape[0]
                else:  # (seq_length, n_leads)
                    sample_seq_length = sample.shape[0]
                    sample_n_leads = sample.shape[1]
                sample_time_axis = np.arange(sample_seq_length)
                
                if is_leads_first:
                    for lead in range(min(n_leads, sample_n_leads)):
                        plt.plot(sample_time_axis, sample[lead], label=f'导联 {lead+1}', color=lead_colors[lead])
                else:
                    for lead in range(min(n_leads, sample_n_leads)):
                        plt.plot(sample_time_axis, sample[:, lead], label=f'导联 {lead+1}', color=lead_colors[lead])
            else:  # 单导联
                sample_time_axis = np.arange(len(sample))
                plt.plot(sample_time_axis, sample, label='导联 1', color=lead_colors[0])
        else:
            sample = x_data[idx]
            if len(sample.shape) == 1:  # 单导联
                plt.plot(time_axis, sample, label='导联 1', color=lead_colors[0])
            else:  # 多导联
                for lead in range(n_leads):
                    if is_leads_first:  # (n_leads, seq_length)
                        plt.plot(time_axis, sample[lead], label=f'导联 {lead+1}', color=lead_colors[lead])
                    else:  # (seq_length, n_leads)
                        plt.plot(time_axis, sample[:, lead], label=f'导联 {lead+1}', color=lead_colors[lead])
        
        plt.title(f'非房颤信号示例 #{i+1}', fontsize=16)
        plt.xlabel('样本点', fontsize=14)
        plt.ylabel('幅值', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_domain', f'non_af_signal_example_{i+1}.png'), dpi=300)
        plt.close()
    
    # 绘制对比图：房颤与非房颤信号
    if len(selected_af) > 0 and len(selected_non_af) > 0:
        plt.figure(figsize=(15, 8))
        
        # 上半部分绘制房颤信号
        plt.subplot(2, 1, 1)
        
        if is_object_array:
            af_sample = x_data[selected_af[0]]
            if len(af_sample.shape) == 2:
                if is_leads_first:  # (n_leads, seq_length)
                    af_signal = af_sample[0]  # 使用第一个导联
                    af_time_axis = np.arange(len(af_signal))
                else:  # (seq_length, n_leads)
                    af_signal = af_sample[:, 0]  # 使用第一个导联
                    af_time_axis = np.arange(len(af_signal))
            else:  # 单导联
                af_signal = af_sample
                af_time_axis = np.arange(len(af_signal))
            plt.plot(af_time_axis, af_signal, color='#e74c3c')
        else:
            af_sample = x_data[selected_af[0]]
            if len(af_sample.shape) == 1:  # 单导联
                plt.plot(time_axis, af_sample, color='#e74c3c')
            else:  # 多导联 (仅显示第一个导联用于对比)
                if is_leads_first:  # (n_leads, seq_length)
                    plt.plot(time_axis, af_sample[0], color='#e74c3c')
                else:  # (seq_length, n_leads)
                    plt.plot(time_axis, af_sample[:, 0], color='#e74c3c')
        
        plt.title('房颤信号示例', fontsize=14)
        plt.ylabel('幅值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 下半部分绘制非房颤信号
        plt.subplot(2, 1, 2)
        
        if is_object_array:
            non_af_sample = x_data[selected_non_af[0]]
            if len(non_af_sample.shape) == 2:
                if is_leads_first:  # (n_leads, seq_length)
                    non_af_signal = non_af_sample[0]  # 使用第一个导联
                    non_af_time_axis = np.arange(len(non_af_signal))
                else:  # (seq_length, n_leads)
                    non_af_signal = non_af_sample[:, 0]  # 使用第一个导联
                    non_af_time_axis = np.arange(len(non_af_signal))
            else:  # 单导联
                non_af_signal = non_af_sample
                non_af_time_axis = np.arange(len(non_af_signal))
            plt.plot(non_af_time_axis, non_af_signal, color='#2ecc71')
        else:
            non_af_sample = x_data[selected_non_af[0]]
            if len(non_af_sample.shape) == 1:  # 单导联
                plt.plot(time_axis, non_af_sample, color='#2ecc71')
            else:  # 多导联 (仅显示第一个导联用于对比)
                if is_leads_first:  # (n_leads, seq_length)
                    plt.plot(time_axis, non_af_sample[0], color='#2ecc71')
                else:  # (seq_length, n_leads)
                    plt.plot(time_axis, non_af_sample[:, 0], color='#2ecc71')
        
        plt.title('非房颤信号示例', fontsize=14)
        plt.xlabel('样本点', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_domain', 'af_vs_non_af_comparison.png'), dpi=300)
        plt.close()
    
    print(f"时域可视化完成: 生成了 {min(num_samples, len(selected_af))} 个房颤示例和 {min(num_samples, len(selected_non_af))} 个非房颤示例")

def analyze_frequency_domain(x_data, y_data, output_dir, sample_rate=250, sample_rates=None, num_samples=3):
    """分析频域特征，支持每个记录使用不同的采样率"""
    print("分析频域特征...")
    
    # 检查是否为对象数组
    is_object_array = isinstance(x_data, np.ndarray) and x_data.dtype == object
    
    # 如果y_data为None，创建一个全零的标签数组
    if y_data is None:
        y_data = np.zeros(len(x_data))
    
    # 选择几个房颤和非房颤样本
    af_indices = np.where(y_data == 1)[0]
    non_af_indices = np.where(y_data == 0)[0]
    
    # 随机选择样本
    np.random.seed(42)  # 设置随机种子以确保可重复性
    selected_af = np.random.choice(af_indices, min(num_samples, len(af_indices)), replace=False) if len(af_indices) > 0 else []
    selected_non_af = np.random.choice(non_af_indices, min(num_samples, len(non_af_indices)), replace=False) if len(non_af_indices) > 0 else []
    
    # 确定数据格式
    if is_object_array:
        print("检测到对象数组，每个样本可能有不同的形状")
        if len(x_data) > 0:
            # 检查第一个样本确定数据排列
            first_sample = x_data[0]
            if len(first_sample.shape) == 2:
                n_leads = first_sample.shape[1]  # 现在排列为(序列长度, 导联数)
                seq_length = first_sample.shape[0]
            else:
                n_leads = 1
                seq_length = len(first_sample)
        else:
            print("警告: 数据为空")
            return
    else:
        # 获取信号长度和导联数
        if x_data.ndim == 3:
            if x_data.shape[1] <= 12:  # 假设为 (样本数, 导联数, 序列长度)
                n_leads = x_data.shape[1]
                seq_length = x_data.shape[2]
                lead_dim = 1
                seq_dim = 2
            else:  # 假设为 (样本数, 序列长度, 导联数)
                n_leads = x_data.shape[2]
                seq_length = x_data.shape[1]
                lead_dim = 2
                seq_dim = 1
        else:  # 单导联 (样本数, 序列长度)
            n_leads = 1
            seq_length = x_data.shape[1]
            lead_dim = None
            seq_dim = 1
    
    # 为导联创建不同颜色
    lead_colors = plt.cm.tab10(np.linspace(0, 1, max(n_leads, 1)))
    
    # 分析房颤样本的频谱
    for i, idx in enumerate(selected_af):
        if i >= num_samples:
            break
            
        # 获取此记录的采样率
        current_sample_rate = sample_rates[idx] if sample_rates else sample_rate
        
        # 获取样本
        sample = x_data[idx]
        
        # 根据数据类型处理
        if is_object_array:
            if len(sample.shape) == 2:  # (序列长度, 导联数)
                sample_seq_length = sample.shape[0]
                sample_n_leads = sample.shape[1]
            else:  # 一维数组
                sample = sample.reshape(-1, 1)  # 转为二维
                sample_seq_length = sample.shape[0]
                sample_n_leads = 1
                
            # 计算频率轴
            frequencies = fftfreq(sample_seq_length, 1/current_sample_rate)
            positive_freq_idx = np.where(frequencies >= 0)[0]
            frequencies = frequencies[positive_freq_idx]
            
            plt.figure(figsize=(15, 6))
            
            # 对每个导联计算FFT
            for lead in range(sample_n_leads):
                signal = sample[:, lead]
                fft_values = fft(signal)
                fft_magnitude = np.abs(fft_values[positive_freq_idx]) / sample_seq_length
                plt.plot(frequencies, fft_magnitude, color=lead_colors[lead % len(lead_colors)], 
                        label=f'导联 {lead+1}')
        else:
            # 使用原始函数逻辑
            frequencies = fftfreq(seq_length, 1/current_sample_rate)
            positive_freq_idx = np.where(frequencies >= 0)[0]
            frequencies = frequencies[positive_freq_idx]
            
            plt.figure(figsize=(15, 6))
            
            if x_data.ndim == 2:  # 单导联
                fft_values = fft(sample)
                fft_magnitude = np.abs(fft_values[positive_freq_idx]) / seq_length
                plt.plot(frequencies, fft_magnitude, color=lead_colors[0], label='导联 1')
            else:  # 多导联
                for lead in range(n_leads):
                    if lead_dim == 1:  # (样本数, 导联数, 序列长度)
                        signal = sample[lead, :]
                    else:  # (样本数, 序列长度, 导联数)
                        signal = sample[:, lead]
                    
                    fft_values = fft(signal)
                    fft_magnitude = np.abs(fft_values[positive_freq_idx]) / seq_length
                    plt.plot(frequencies, fft_magnitude, color=lead_colors[lead], 
                            label=f'导联 {lead+1}')
        
        plt.title(f'房颤信号频谱分析 #{i+1} (采样率: {current_sample_rate} Hz)', fontsize=16)
        plt.xlabel('频率 (Hz)', fontsize=14)
        plt.ylabel('幅度', fontsize=14)
        plt.xlim(0, min(40, current_sample_rate/2))  # 限制频率范围，最高显示到奈奎斯特频率
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_domain', f'af_spectrum_{i+1}.png'), dpi=300)
        plt.close()
    
    # 分析非房颤样本的频谱
    for i, idx in enumerate(selected_non_af):
        if i >= num_samples:
            break
            
        # 获取此记录的采样率
        current_sample_rate = sample_rates[idx] if sample_rates else sample_rate
        
        # 获取样本
        sample = x_data[idx]
        
        # 根据数据类型处理
        if is_object_array:
            if len(sample.shape) == 2:  # (序列长度, 导联数)
                sample_seq_length = sample.shape[0]
                sample_n_leads = sample.shape[1]
            else:  # 一维数组
                sample = sample.reshape(-1, 1)  # 转为二维
                sample_seq_length = sample.shape[0]
                sample_n_leads = 1
                
            # 计算频率轴
            frequencies = fftfreq(sample_seq_length, 1/current_sample_rate)
            positive_freq_idx = np.where(frequencies >= 0)[0]
            frequencies = frequencies[positive_freq_idx]
            
            plt.figure(figsize=(15, 6))
            
            # 对每个导联计算FFT
            for lead in range(sample_n_leads):
                signal = sample[:, lead]
                fft_values = fft(signal)
                fft_magnitude = np.abs(fft_values[positive_freq_idx]) / sample_seq_length
                plt.plot(frequencies, fft_magnitude, color=lead_colors[lead % len(lead_colors)], 
                        label=f'导联 {lead+1}')
        else:
            # 使用原始函数逻辑
            frequencies = fftfreq(seq_length, 1/current_sample_rate)
            positive_freq_idx = np.where(frequencies >= 0)[0]
            frequencies = frequencies[positive_freq_idx]
            
            plt.figure(figsize=(15, 6))
            
            if x_data.ndim == 2:  # 单导联
                fft_values = fft(sample)
                fft_magnitude = np.abs(fft_values[positive_freq_idx]) / seq_length
                plt.plot(frequencies, fft_magnitude, color=lead_colors[0], label='导联 1')
            else:  # 多导联
                for lead in range(n_leads):
                    if lead_dim == 1:  # (样本数, 导联数, 序列长度)
                        signal = sample[lead, :]
                    else:  # (样本数, 序列长度, 导联数)
                        signal = sample[:, lead]
                    
                    fft_values = fft(signal)
                    fft_magnitude = np.abs(fft_values[positive_freq_idx]) / seq_length
                    plt.plot(frequencies, fft_magnitude, color=lead_colors[lead], 
                            label=f'导联 {lead+1}')
        
        plt.title(f'非房颤信号频谱分析 #{i+1} (采样率: {current_sample_rate} Hz)', fontsize=16)
        plt.xlabel('频率 (Hz)', fontsize=14)
        plt.ylabel('幅度', fontsize=14)
        plt.xlim(0, min(40, current_sample_rate/2))  # 限制频率范围，最高显示到奈奎斯特频率
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_domain', f'non_af_spectrum_{i+1}.png'), dpi=300)
        plt.close()
    
    # 简化后续代码 - 仅在有足够样本时创建对比图
    if len(selected_af) > 0 and len(selected_non_af) > 0:
        print("创建房颤与非房颤信号频谱对比图")
        
        # 仅创建简单的对比图，显示第一个AF和非AF样本的第一个导联
        plt.figure(figsize=(15, 8))
        
        # 处理房颤样本
        af_idx = selected_af[0]
        af_sample = x_data[af_idx]
        af_sample_rate = sample_rates[af_idx] if sample_rates else sample_rate
        
        plt.subplot(2, 1, 1)
        
        if is_object_array:
            if len(af_sample.shape) == 2:
                af_signal = af_sample[:, 0]  # 使用第一个导联
                af_seq_length = af_sample.shape[0]
            else:
                af_signal = af_sample
                af_seq_length = len(af_signal)
                
            # 计算频率轴和FFT
            af_frequencies = fftfreq(af_seq_length, 1/af_sample_rate)
            af_positive_freq_idx = np.where(af_frequencies >= 0)[0]
            af_frequencies = af_frequencies[af_positive_freq_idx]
            
            fft_values = fft(af_signal)
            fft_magnitude = np.abs(fft_values[af_positive_freq_idx]) / af_seq_length
            plt.plot(af_frequencies, fft_magnitude, color='#e74c3c')
        else:
            # 使用原来的逻辑
            af_frequencies = fftfreq(seq_length, 1/af_sample_rate)
            af_positive_freq_idx = np.where(af_frequencies >= 0)[0]
            af_frequencies = af_frequencies[af_positive_freq_idx]
            
            if x_data.ndim == 2:  # 单导联
                fft_values = fft(af_sample)
                fft_magnitude = np.abs(fft_values[af_positive_freq_idx]) / seq_length
                plt.plot(af_frequencies, fft_magnitude, color='#e74c3c')
            else:  # 多导联
                if lead_dim == 1:  # (样本数, 导联数, 序列长度)
                    signal = af_sample[0, :]
                else:  # (样本数, 序列长度, 导联数)
                    signal = af_sample[:, 0]
                    
                fft_values = fft(signal)
                fft_magnitude = np.abs(fft_values[af_positive_freq_idx]) / seq_length
                plt.plot(af_frequencies, fft_magnitude, color='#e74c3c')
        
        plt.title(f'房颤信号频谱 (采样率: {af_sample_rate} Hz)', fontsize=14)
        plt.ylabel('幅度', fontsize=12)
        plt.xlim(0, min(40, af_sample_rate/2))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 处理非房颤样本
        non_af_idx = selected_non_af[0]
        non_af_sample = x_data[non_af_idx]
        non_af_sample_rate = sample_rates[non_af_idx] if sample_rates else sample_rate
        
        plt.subplot(2, 1, 2)
        
        if is_object_array:
            if len(non_af_sample.shape) == 2:
                non_af_signal = non_af_sample[:, 0]  # 使用第一个导联
                non_af_seq_length = non_af_sample.shape[0]
            else:
                non_af_signal = non_af_sample
                non_af_seq_length = len(non_af_signal)
                
            # 计算频率轴和FFT
            non_af_frequencies = fftfreq(non_af_seq_length, 1/non_af_sample_rate)
            non_af_positive_freq_idx = np.where(non_af_frequencies >= 0)[0]
            non_af_frequencies = non_af_frequencies[non_af_positive_freq_idx]
            
            fft_values = fft(non_af_signal)
            fft_magnitude = np.abs(fft_values[non_af_positive_freq_idx]) / non_af_seq_length
            plt.plot(non_af_frequencies, fft_magnitude, color='#2ecc71')
        else:
            # 使用原来的逻辑
            non_af_frequencies = fftfreq(seq_length, 1/non_af_sample_rate)
            non_af_positive_freq_idx = np.where(non_af_frequencies >= 0)[0]
            non_af_frequencies = non_af_frequencies[non_af_positive_freq_idx]
            
            if x_data.ndim == 2:  # 单导联
                fft_values = fft(non_af_sample)
                fft_magnitude = np.abs(fft_values[non_af_positive_freq_idx]) / seq_length
                plt.plot(non_af_frequencies, fft_magnitude, color='#2ecc71')
            else:  # 多导联
                if lead_dim == 1:  # (样本数, 导联数, 序列长度)
                    signal = non_af_sample[0, :]
                else:  # (样本数, 序列长度, 导联数)
                    signal = non_af_sample[:, 0]
                    
                fft_values = fft(signal)
                fft_magnitude = np.abs(fft_values[non_af_positive_freq_idx]) / seq_length
                plt.plot(non_af_frequencies, fft_magnitude, color='#2ecc71')
        
        plt.title(f'非房颤信号频谱 (采样率: {non_af_sample_rate} Hz)', fontsize=14)
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('幅度', fontsize=12)
        plt.xlim(0, min(40, non_af_sample_rate/2))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_domain', 'af_vs_non_af_spectrum_comparison.png'), dpi=300)
        plt.close()
    
    print("频域分析完成")

def generate_summary_report(output_dir):
    """生成汇总报告"""
    print("生成汇总报告...")
    
    # 收集所有生成的图
    report_content = ["# ECG原始数据集分布分析报告\n", 
                      "## 1. 原始数据特性\n"]
    
    # 添加原始数据特性图
    raw_data_files = [f for f in os.listdir(os.path.join(output_dir, 'raw_data')) 
                     if f.endswith('.png')]
    for img_file in sorted(raw_data_files):
        img_path = os.path.join('raw_data', img_file)
        report_content.append(f"![{img_file}]({img_path})\n")
    
    # 尝试加载JSON摘要文件并添加到报告
    json_path = os.path.join(output_dir, 'raw_data', 'data_summary.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                summary = json.load(f)
            
            report_content.append("\n### 数据摘要\n")
            report_content.append(f"- 样本数量: {summary.get('n_samples', 'N/A')}\n")
            report_content.append(f"- 导联数量: {summary.get('n_leads', 'N/A')}\n")
            report_content.append(f"- 数据形状: {summary.get('data_shape', 'N/A')}\n")
            
            if 'sampling_rates' in summary:
                sr = summary['sampling_rates']
                report_content.append("\n#### 采样率统计\n")
                report_content.append(f"- 最小采样率: {sr.get('min', 'N/A')} Hz\n")
                report_content.append(f"- 最大采样率: {sr.get('max', 'N/A')} Hz\n")
                report_content.append(f"- 平均采样率: {sr.get('mean', 'N/A')} Hz\n")
                
                if 'distribution' in sr:
                    report_content.append("\n采样率分布:\n")
                    for rate, count in sr['distribution'].items():
                        report_content.append(f"- {rate} Hz: {count} 条记录\n")
            
            if 'record_lengths' in summary:
                rl = summary['record_lengths']
                report_content.append("\n#### 记录长度统计\n")
                report_content.append(f"- 最短记录: {rl.get('min', 'N/A')} 样本点\n")
                report_content.append(f"- 最长记录: {rl.get('max', 'N/A')} 样本点\n")
                report_content.append(f"- 平均长度: {rl.get('mean', 'N/A')} 样本点\n")
                report_content.append(f"- 中位数长度: {rl.get('median', 'N/A')} 样本点\n")
        except:
            pass
    
    # 添加类别分布(如果存在)
    class_dist_path = os.path.join(output_dir, 'class_distribution')
    if os.path.exists(class_dist_path):
        class_dist_files = [f for f in os.listdir(class_dist_path) if f.endswith('.png')]
        if class_dist_files:
            report_content.append("\n## 2. 类别分布\n")
            for img_file in sorted(class_dist_files):
                img_path = os.path.join('class_distribution', img_file)
                report_content.append(f"![{img_file}]({img_path})\n")
    
    # 添加信号特征(如果存在)
    signal_feature_path = os.path.join(output_dir, 'signal_features')
    if os.path.exists(signal_feature_path):
        signal_feature_files = [f for f in os.listdir(signal_feature_path) if f.endswith('.png')]
        if signal_feature_files:
            report_content.append("\n## 3. 信号特征分布\n")
            for img_file in sorted(signal_feature_files):
                img_path = os.path.join('signal_features', img_file)
                report_content.append(f"![{img_file}]({img_path})\n")
    
    # 添加时域信号示例(如果存在)
    time_domain_path = os.path.join(output_dir, 'time_domain')
    if os.path.exists(time_domain_path):
        time_domain_files = [f for f in os.listdir(time_domain_path) if f.endswith('.png')]
        if time_domain_files:
            report_content.append("\n## 4. 时域信号示例\n")
            for img_file in sorted(time_domain_files):
                img_path = os.path.join('time_domain', img_file)
                report_content.append(f"![{img_file}]({img_path})\n")
    
    # 添加频域分析结果(如果存在)
    freq_domain_path = os.path.join(output_dir, 'frequency_domain')
    if os.path.exists(freq_domain_path):
        freq_domain_files = [f for f in os.listdir(freq_domain_path) if f.endswith('.png')]
        if freq_domain_files:
            report_content.append("\n## 5. 频域分析\n")
            for img_file in sorted(freq_domain_files):
                img_path = os.path.join('frequency_domain', img_file)
                report_content.append(f"![{img_file}]({img_path})\n")
    
    # 写入报告文件
    with open(os.path.join(output_dir, 'raw_data_analysis_report.md'), 'w', encoding='utf-8') as f:
        f.writelines(report_content)
    
    print(f"汇总报告已生成: {os.path.join(output_dir, 'raw_data_analysis_report.md')}")

def analyze_raw_data_properties(x_data, sample_rates, record_lengths, output_dir):
    """分析原始数据的特性，包括采样率、记录长度分布等"""
    print("分析原始数据特性...")
    
    # 创建基本信息摘要
    n_samples = len(x_data)
    
    # 检查数据类型和形状
    if isinstance(x_data, np.ndarray) and x_data.dtype == object:
        print("检测到异构数据数组(dtype=object)，将分析每个样本的独立形状")
        # 确定数据维度和导联数
        sample_shapes = [x.shape for x in x_data]
        print(f"样本形状示例: {sample_shapes[:3]}...")
        
        # 尝试确定导联数量
        if len(sample_shapes) > 0:
            # 检查第一个样本确定数据排列方式
            first_sample = x_data[0]
            if len(first_sample.shape) == 2:  # (导联数, 序列长度) 或 (序列长度, 导联数)
                # 假设导联数是较小的维度
                n_leads = min(first_sample.shape)
                seq_length_dim = 0 if first_sample.shape[0] > first_sample.shape[1] else 1
                lead_dim = 1 if seq_length_dim == 0 else 0
                n_leads = first_sample.shape[lead_dim]
            else:
                n_leads = 1  # 单导联
                seq_length_dim = 0
    else:
        # 获取导联数量和序列长度维度
        if x_data.ndim == 3:
            if x_data.shape[1] <= 12:  # 假设为 (样本数, 导联数, 序列长度)
                n_leads = x_data.shape[1]
                seq_length_dim = 2
                lead_dim = 1
            else:  # 假设为 (样本数, 序列长度, 导联数)
                n_leads = x_data.shape[2]
                seq_length_dim = 1
                lead_dim = 2
        else:  # 单导联 (样本数, 序列长度)
            n_leads = 1
            seq_length_dim = 1
            lead_dim = None
    
    # 获取数据形状描述
    if isinstance(x_data, np.ndarray) and x_data.dtype == object:
        data_shape = "异构数组，每个样本可能有不同形状"
    else:
        data_shape = str(x_data.shape)
    
    # 打印基本信息
    print(f"样本数量: {n_samples}")
    print(f"导联数量: {n_leads}")
    print(f"数据形状: {data_shape}")
    
    # 分析采样率分布
    if sample_rates:
        unique_rates = np.unique(sample_rates)
        rate_counts = {rate: sample_rates.count(rate) for rate in unique_rates}
        
        print("采样率分布:")
        for rate, count in rate_counts.items():
            print(f"  - {rate} Hz: {count} 条记录 ({count/n_samples*100:.2f}%)")
        
        # 绘制采样率分布图
        plt.figure(figsize=(10, 6))
        rates = list(rate_counts.keys())
        counts = list(rate_counts.values())
        bars = plt.bar(rates, counts, color='#3498db')
        plt.title('数据采样率分布', fontsize=16)
        plt.xlabel('采样率 (Hz)', fontsize=14)
        plt.ylabel('记录数量', fontsize=14)
        
        # 为条形图添加标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}\n({height/sum(counts):.1%})',
                    ha='center', va='bottom', fontsize=12)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'raw_data', 'sampling_rate_distribution.png'), dpi=300)
        plt.close()
    
    # 分析记录长度分布
    if record_lengths:
        # 计算长度统计
        min_length = min(record_lengths)
        max_length = max(record_lengths)
        mean_length = np.mean(record_lengths)
        median_length = np.median(record_lengths)
        
        print(f"记录长度统计:")
        print(f"  - 最小长度: {min_length} 样本点")
        print(f"  - 最大长度: {max_length} 样本点")
        print(f"  - 平均长度: {mean_length:.2f} 样本点")
        print(f"  - 中位数长度: {median_length} 样本点")
        
        # 绘制记录长度直方图
        plt.figure(figsize=(12, 6))
        plt.hist(record_lengths, bins=30, color='#2ecc71', alpha=0.7)
        plt.axvline(mean_length, color='#e74c3c', linestyle='--', 
                   label=f'平均值: {mean_length:.2f}')
        plt.axvline(median_length, color='#3498db', linestyle='-', 
                   label=f'中位数: {median_length}')
        plt.title('记录长度分布', fontsize=16)
        plt.xlabel('记录长度 (样本点)', fontsize=14)
        plt.ylabel('记录数量', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'raw_data', 'record_length_distribution.png'), dpi=300)
        plt.close()
        
        # 如果长度不一致，分析长度变化对信号的影响
        if min_length != max_length:
            # 找到最短和最长的几个记录
            sorted_indices = np.argsort(record_lengths)
            shortest_indices = sorted_indices[:3]
            longest_indices = sorted_indices[-3:]
            
            # 绘制最短记录示例
            plt.figure(figsize=(15, 8))
            plt.suptitle('最短记录示例', fontsize=16)
            
            for i, idx in enumerate(shortest_indices):
                if i >= 3:  # 最多绘制3个样本
                    break
                    
                if isinstance(x_data, np.ndarray) and x_data.dtype == object:
                    sample = x_data[idx]
                    # 对于object数组，直接使用第一个导联
                    if len(sample.shape) == 2:
                        signal = sample[0, :] if sample.shape[0] <= 12 else sample[:, 0]
                    else:
                        signal = sample
                else:
                    sample = x_data[idx]
                    if x_data.ndim == 2:
                        signal = sample
                    else:
                        # 根据导联维度选择第一个导联
                        if seq_length_dim == 2:  # (样本数, 导联数, 序列长度)
                            signal = sample[0, :]
                        else:  # (样本数, 序列长度, 导联数)
                            signal = sample[:, 0]
                
                plt.subplot(3, 1, i+1)
                plt.plot(signal)
                plt.title(f'记录 #{idx+1}, 长度: {record_lengths[idx]} 样本点')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join(output_dir, 'raw_data', 'shortest_records.png'), dpi=300)
            plt.close()
            
            # 绘制最长记录示例
            plt.figure(figsize=(15, 8))
            plt.suptitle('最长记录示例', fontsize=16)
            
            for i, idx in enumerate(longest_indices):
                if i >= 3:  # 最多绘制3个样本
                    break
                    
                if isinstance(x_data, np.ndarray) and x_data.dtype == object:
                    sample = x_data[idx]
                    # 对于object数组，直接使用第一个导联
                    if len(sample.shape) == 2:
                        signal = sample[0, :] if sample.shape[0] <= 12 else sample[:, 0]
                    else:
                        signal = sample
                else:
                    sample = x_data[idx]
                    if x_data.ndim == 2:
                        signal = sample
                    else:
                        # 根据导联维度选择第一个导联
                        if seq_length_dim == 2:  # (样本数, 导联数, 序列长度)
                            signal = sample[0, :]
                        else:  # (样本数, 序列长度, 导联数)
                            signal = sample[:, 0]
                
                plt.subplot(3, 1, i+1)
                plt.plot(signal)
                plt.title(f'记录 #{idx+1}, 长度: {record_lengths[idx]} 样本点')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join(output_dir, 'raw_data', 'longest_records.png'), dpi=300)
            plt.close()
    
    # 保存原始数据分析摘要
    summary = {
        'n_samples': n_samples,
        'n_leads': n_leads,
        'data_shape': data_shape,
    }
    
    if sample_rates:
        summary['sampling_rates'] = {
            'min': float(min(sample_rates)),
            'max': float(max(sample_rates)),
            'mean': float(np.mean(sample_rates)),
            'unique_values': [float(r) for r in unique_rates],
            'distribution': {str(r): c for r, c in rate_counts.items()}
        }
    
    if record_lengths:
        summary['record_lengths'] = {
            'min': int(min_length),
            'max': int(max_length),
            'mean': float(mean_length),
            'median': float(median_length)
        }
    
    # 保存为JSON文件
    with open(os.path.join(output_dir, 'raw_data', 'data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("原始数据特性分析完成")

def main():
    parser = argparse.ArgumentParser(description='ECG原始数据集分布分析工具')
    parser.add_argument('--data_path', type=str, required=True, help='原始数据文件或目录路径')
    parser.add_argument('--labels_path', type=str, help='标签文件路径(可选)')
    parser.add_argument('--max_seq_length', type=int, help='最大序列长度(可选)')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis', help='输出目录')
    parser.add_argument('--sample_rate', type=int, default=250, help='默认采样率(Hz)，当原始数据未提供采样率时使用')
    parser.add_argument('--max_samples', type=int, help='要分析的最大样本数(可选)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_output_dir(args.output_dir)
    
    # 打印开始分析信息
    print(f"开始分析ECG原始数据: {args.data_path}")
    print(f"标签路径: {args.labels_path if args.labels_path else '未提供'}")
    
    try:
        # 加载原始数据
        result = load_raw_data(args.data_path, args.labels_path, args.max_seq_length)
        
        if len(result) == 4:
            x_data, y_data, sample_rates, record_lengths = result
        else:
            x_data, y_data = result
            sample_rates = None
            record_lengths = None
        
        # 如果指定了最大样本数，则截断数据
        if args.max_samples and args.max_samples < len(x_data):
            indices = np.random.choice(len(x_data), args.max_samples, replace=False)
            x_data = x_data[indices]
            if y_data is not None:
                y_data = y_data[indices]
            if sample_rates is not None and isinstance(sample_rates, np.ndarray):
                sample_rates = sample_rates[indices]
            if record_lengths is not None and isinstance(record_lengths, np.ndarray):
                record_lengths = record_lengths[indices]
        
        # 打印加载的数据形状
        print(f"数据形状: {x_data.shape}")
        if y_data is not None:
            print(f"标签形状: {y_data.shape}")
            print(f"阳性率: {np.mean(y_data):.4f}")
        
        # 分析原始数据特性
        analyze_raw_data_properties(x_data, sample_rates, record_lengths, output_dir)
        
        # 如果有标签，进行类别分布分析
        if y_data is not None:
            analyze_class_distribution(y_data, output_dir)
            analyze_signal_features(x_data, y_data, output_dir)
            visualize_time_domain(x_data, y_data, output_dir, num_samples=5)
            
            # 使用默认或检测到的采样率进行频域分析
            if sample_rates is not None:
                analyze_frequency_domain(x_data, y_data, output_dir, sample_rates=sample_rates, num_samples=3)
            else:
                analyze_frequency_domain(x_data, y_data, output_dir, sample_rate=args.sample_rate, num_samples=3)
        else:
            # 无标签情况下的可视化(不区分类别)
            visualize_time_domain(x_data, None, output_dir, num_samples=5)
            
            # 使用默认或检测到的采样率进行频域分析
            if sample_rates is not None:
                analyze_frequency_domain(x_data, None, output_dir, sample_rates=sample_rates, num_samples=3)
            else:
                analyze_frequency_domain(x_data, None, output_dir, sample_rate=args.sample_rate, num_samples=3)
        
        # 生成汇总报告
        generate_summary_report(output_dir)
        
        print(f"分析完成! 报告保存在: {output_dir}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"分析过程中发生错误: {str(e)}")
        print("请检查输入数据路径和格式是否正确。")

if __name__ == "__main__":
    main()