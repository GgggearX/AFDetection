import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_training_history(rnn_history, densenet_history, metric='loss'):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 6))
    
    # RNN训练历史
    plt.plot(rnn_history[metric], 'b-', label=f'RNN Training {metric}')
    plt.plot(rnn_history[f'val_{metric}'], 'b--', label=f'RNN Validation {metric}')
    
    # DenseNet训练历史
    plt.plot(densenet_history[metric], 'r-', label=f'DenseNet Training {metric}')
    plt.plot(densenet_history[f'val_{metric}'], 'r--', label=f'DenseNet Validation {metric}')
    
    plt.title(f'Model {metric.capitalize()} History')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/training_{metric}_history.png')
    plt.close()

def plot_prediction_distribution(predictions_df):
    """绘制预测分布"""
    plt.figure(figsize=(15, 5))
    
    # 创建子图
    for i, model in enumerate(['RNN_Prediction', 'DenseNet_Prediction', 'Ensemble_Prediction']):
        plt.subplot(1, 3, i+1)
        
        # 分别绘制正负样本的预测分布
        for label in [0, 1]:
            mask = predictions_df['True_Label'] == label
            sns.kdeplot(predictions_df[model][mask], 
                       label=f'Label {label}',
                       fill=True)
        
        plt.title(f'{model.split("_")[0]} Predictions')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/prediction_distributions.png')
    plt.close()

def plot_confusion_matrices(predictions_df, threshold=0.5):
    """绘制混淆矩阵"""
    plt.figure(figsize=(15, 5))
    
    for i, model in enumerate(['RNN_Prediction', 'DenseNet_Prediction', 'Ensemble_Prediction']):
        plt.subplot(1, 3, i+1)
        
        # 计算混淆矩阵
        y_pred = (predictions_df[model] > threshold).astype(int)
        cm = confusion_matrix(predictions_df['True_Label'], y_pred)
        
        # 绘制混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model.split("_")[0]} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png')
    plt.close()

def plot_fold_comparison(fold_results):
    """绘制不同fold之间的性能比较"""
    plt.figure(figsize=(12, 6))
    
    metrics = ['accuracy', 'auc']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        
        data = []
        labels = []
        for model in ['RNN', 'DenseNet']:
            for fold in range(1, 6):
                data.append(fold_results[f'{model}_fold_{fold}'][metric])
                labels.extend([f'{model} Fold {fold}'])
        
        plt.bar(range(len(data)), data)
        plt.xticks(range(len(data)), labels, rotation=45)
        plt.title(f'{metric.upper()} Comparison Across Folds')
        plt.ylabel(metric.capitalize())
    
    plt.tight_layout()
    plt.savefig('outputs/fold_comparison.png')
    plt.close()

def create_summary_table(predictions_df, threshold=0.5):
    """创建模型性能总结表"""
    summary = []
    
    for model in ['RNN_Prediction', 'DenseNet_Prediction', 'Ensemble_Prediction']:
        y_pred = (predictions_df[model] > threshold).astype(int)
        y_true = predictions_df['True_Label']
        
        # 计算各种指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        summary.append({
            'Model': model.split('_')[0],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('outputs/model_performance_summary.csv', index=False)
    return summary_df

def main():
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 加载预测结果
    predictions_df = pd.read_csv('outputs/predictions.csv')
    
    # 加载训练历史（假设已保存）
    rnn_history = pd.read_csv('RNN/logs/fold_1/training.log')
    densenet_history = pd.read_csv('DenseNet/logs/fold_1/training.log')
    
    # 绘制各种可视化
    plot_training_history(rnn_history, densenet_history, 'loss')
    plot_training_history(rnn_history, densenet_history, 'accuracy')
    plot_prediction_distribution(predictions_df)
    plot_confusion_matrices(predictions_df)
    
    # 创建性能总结
    summary_df = create_summary_table(predictions_df)
    print("\n模型性能总结:")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()

# 执行命令：
# python visualize_results.py 