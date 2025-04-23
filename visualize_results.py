import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
import os
import matplotlib as mpl

# Set global font and style parameters
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans', 'Helvetica']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.autolayout'] = True
plt.style.use('seaborn')

def plot_training_history(model_dirs):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 8))
    
    # 定义颜色
    colors = ['b', 'g', 'r', 'c', 'm']
    has_valid_data = False
    
    for i, model_dir in enumerate(model_dirs):
        color = colors[i % len(colors)]
        label = os.path.basename(model_dir)
        print(f"处理模型: {label}")
        
        # 读取所有fold的训练历史
        for fold in range(1, 6):
            # 尝试不同的可能文件路径
            possible_paths = [
                os.path.join(model_dir, 'logs', f'fold_{fold}', 'training_history.csv'),
                os.path.join(model_dir, 'logs', f'fold_{fold}', f'training_fold_{fold}.csv'),
                os.path.join(model_dir, 'logs', f'training_fold_{fold}.csv')
            ]
            
            history_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    history_path = path
                    break
            
            if history_path:
                print(f"  读取 fold_{fold} 的训练历史: {history_path}")
                try:
                    history = pd.read_csv(history_path)
                    print(f"  可用的列: {', '.join(history.columns)}")
                    
                    # 处理不同的列名
                    auc_cols = [col for col in history.columns if 'auc' in col.lower()]
                    val_auc_cols = [col for col in auc_cols if 'val' in col.lower()]
                    train_auc_cols = [col for col in auc_cols if 'val' not in col.lower()]
                    
                    if train_auc_cols and val_auc_cols:
                        train_auc_col = train_auc_cols[0]
                        val_auc_col = val_auc_cols[0]
                        
                        plt.plot(history[train_auc_col], color=color, alpha=0.3, 
                                label=f'{label} Fold {fold} (训练)')
                        plt.plot(history[val_auc_col], color=color, alpha=0.3, 
                                linestyle='--', label=f'{label} Fold {fold} (验证)')
                        has_valid_data = True
                    else:
                        print(f"  警告: 在{history_path}中未找到AUC相关的列")
                except Exception as e:
                    print(f"  错误: 读取{history_path}时出错: {str(e)}")
            else:
                print(f"  警告: 未找到fold_{fold}的训练历史文件")
    
    if has_valid_data:
        plt.title('训练历史 - AUC')
        plt.xlabel('轮次')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join('outputs', 'training_history.png')
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史曲线已保存到: {save_path}")
    else:
        plt.close()
        print("警告: 没有找到有效的训练历史数据，跳过绘图")

def plot_roc_curves(model_dirs):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm']
    has_valid_data = False
    
    for i, model_dir in enumerate(model_dirs):
        color = colors[i % len(colors)]
        label = os.path.basename(model_dir)
        print(f"处理模型: {label}")
        
        # 处理每个fold的ROC数据
        for fold in range(1, 6):
            # 尝试不同的可能文件路径
            possible_paths = [
                os.path.join(model_dir, 'logs', f'fold_{fold}', 'roc_data.csv'),
                os.path.join(model_dir, 'logs', f'fold_{fold}', f'roc_fold_{fold}.csv'),
                os.path.join(model_dir, 'logs', f'roc_fold_{fold}.csv')
            ]
            
            roc_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    roc_path = path
                    break
            
            if roc_path:
                print(f"  读取 fold_{fold} 的ROC数据: {roc_path}")
                try:
                    roc_data = pd.read_csv(roc_path)
                    print(f"  可用的列: {', '.join(roc_data.columns)}")
                    
                    # 检查必要的列
                    required_cols = ['fpr', 'tpr']
                    if all(col in roc_data.columns for col in required_cols):
                        fpr = roc_data['fpr'].values
                        tpr = roc_data['tpr'].values
                        
                        # 计算AUC
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, color=color, alpha=0.3,
                                label=f'{label} Fold {fold} (AUC = {roc_auc:.3f})')
                        has_valid_data = True
                    else:
                        print(f"  警告: 在{roc_path}中未找到必要的列 (fpr, tpr)")
                except Exception as e:
                    print(f"  错误: 读取{roc_path}时出错: {str(e)}")
                    if 'roc_data' in locals():
                        print(f"  文件内容预览:\n{roc_data.head()}")
            else:
                print(f"  警告: 未找到fold_{fold}的ROC数据文件")
    
    if has_valid_data:
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True)
        
        save_path = os.path.join('outputs', 'roc_curves.png')
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC曲线已保存到: {save_path}")
    else:
        plt.close()
        print("警告: 没有找到有效的ROC数据，跳过绘图")

def plot_confusion_matrices(model_dirs):
    """绘制混淆矩阵"""
    n_models = len(model_dirs)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, (model_dir, ax) in enumerate(zip(model_dirs, axes)):
        label = os.path.basename(model_dir)
        print(f"处理模型: {label}")
        
        # 读取所有fold的混淆矩阵并求平均
        cm_sum = None
        fold_count = 0
        for fold in range(1, 6):
            # 尝试不同的可能文件路径
            possible_paths = [
                os.path.join(model_dir, 'results', f'fold_{fold}', 'confusion_matrix.csv'),
                os.path.join(model_dir, 'results', f'confusion_matrix_fold_{fold}.csv')
            ]
            
            cm_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    cm_path = path
                    break
            
            if cm_path:
                print(f"  读取 fold_{fold} 的混淆矩阵")
                try:
                    cm = pd.read_csv(cm_path, index_col=0).values
                    if cm_sum is None:
                        cm_sum = cm
                    else:
                        cm_sum += cm
                    fold_count += 1
                except Exception as e:
                    print(f"  错误: 读取{cm_path}时出错: {str(e)}")
            else:
                print(f"  警告: 未找到fold_{fold}的混淆矩阵文件")
        
        if cm_sum is not None and fold_count > 0:
            cm_avg = cm_sum / fold_count
            sns.heatmap(cm_avg, annot=True, fmt='.2f', cmap='Blues', ax=ax)
            ax.set_title(f'{label}混淆矩阵')
            ax.set_xlabel('预测值')
            ax.set_ylabel('真实值')
    
    plt.tight_layout()
    save_path = os.path.join('outputs', 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")

def main():
    # 创建输出目录
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 模型目录列表
    model_dirs = [
        'RNN',
        'DenseNet',
        'CNN-LSTM',
        'WaveNet',  # 修正WaveNet的路径
        'Transformer'
    ]
    
    # 检查每个模型目录是否存在
    print("\n检查模型目录:")
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"✓ {model_dir} 存在")
        else:
            print(f"✗ {model_dir} 不存在")
    
    # 生成可视化结果
    print("\n开始生成可视化结果...")
    
    # 训练历史曲线
    print("\n1. 生成训练历史曲线...")
    plot_training_history(model_dirs)
    
    # ROC曲线
    print("\n2. 生成ROC曲线...")
    plot_roc_curves(model_dirs)
    
    # 混淆矩阵
    print("\n3. 生成混淆矩阵...")
    plot_confusion_matrices(model_dirs)
    
    # 检查生成的文件
    print("\n检查生成的文件:")
    for filename in ['training_history.png', 'roc_curves.png', 'confusion_matrices.png']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} 已生成")
        else:
            print(f"✗ {filename} 未生成")
    
    print(f"\n所有可视化结果已保存到: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()

# 执行命令：
# python visualize_results.py 