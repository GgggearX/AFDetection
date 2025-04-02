import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
import os

# Set global font and style parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'figure.autolayout': True
})
plt.style.use('seaborn')

def plot_training_history(rnn_history, densenet_history, cnn_lstm_history, wavenet_history, metric='loss'):
    """Plot training history curves"""
    plt.figure(figsize=(12, 6))
    
    # RNN training history
    plt.plot(rnn_history[metric], 'b-', label=f'RNN Training {metric}')
    plt.plot(rnn_history[f'val_{metric}'], 'b--', label=f'RNN Validation {metric}')
    
    # DenseNet training history
    plt.plot(densenet_history[metric], 'r-', label=f'DenseNet Training {metric}')
    plt.plot(densenet_history[f'val_{metric}'], 'r--', label=f'DenseNet Validation {metric}')
    
    # CNN-LSTM training history
    plt.plot(cnn_lstm_history[metric], 'g-', label=f'CNN-LSTM Training {metric}')
    plt.plot(cnn_lstm_history[f'val_{metric}'], 'g--', label=f'CNN-LSTM Validation {metric}')
    
    # WaveNet training history
    plt.plot(wavenet_history[metric], 'm-', label=f'WaveNet Training {metric}')
    plt.plot(wavenet_history[f'val_{metric}'], 'm--', label=f'WaveNet Validation {metric}')
    
    plt.title(f'Model {metric.capitalize()} History')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/training_{metric}_history.png')
    plt.close()

def plot_prediction_distribution(predictions_df):
    """Plot prediction distribution for all models"""
    plt.figure(figsize=(15, 5))
    
    # Create subplots
    for i, model in enumerate(['RNN_Prediction', 'DenseNet_Prediction', 'CNN-LSTM_Prediction', 'WaveNet_Prediction', 'Ensemble_Prediction']):
        plt.subplot(1, 5, i+1)
        
        # Plot distribution for positive and negative samples
        for label in [0, 1]:
            mask = predictions_df['True_Label'] == label
            sns.kdeplot(predictions_df[model][mask], 
                       label=f'Class {label}',
                       fill=True)
        
        plt.title(f'{model.split("_")[0]} Predictions')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/prediction_distributions.png')
    plt.close()

def plot_confusion_matrices(predictions_df, threshold=0.5):
    """Plot confusion matrices for all models"""
    plt.figure(figsize=(15, 5))
    
    for i, model in enumerate(['RNN_Prediction', 'DenseNet_Prediction', 'CNN-LSTM_Prediction', 'WaveNet_Prediction', 'Ensemble_Prediction']):
        plt.subplot(1, 5, i+1)
        
        # Calculate confusion matrix
        y_pred = (predictions_df[model] > threshold).astype(int)
        cm = confusion_matrix(predictions_df['True_Label'], y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model.split("_")[0]} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png')
    plt.close()

def plot_fold_comparison(fold_results):
    """Plot performance comparison between different folds"""
    plt.figure(figsize=(12, 6))
    
    metrics = ['accuracy', 'auc']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        
        data = []
        labels = []
        for model in ['RNN', 'DenseNet', 'CNN-LSTM', 'WaveNet']:
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
    """Create model performance summary table"""
    summary = []
    
    for model in ['RNN_Prediction', 'DenseNet_Prediction', 'CNN-LSTM_Prediction', 'WaveNet_Prediction', 'Ensemble_Prediction']:
        y_pred = (predictions_df[model] > threshold).astype(int)
        y_true = predictions_df['True_Label']
        
        # Calculate metrics
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

def plot_model_predictions(predictions_df, model_name, threshold=0.5):
    """Plot prediction results for a single model"""
    plt.figure(figsize=(12, 6))
    
    # Plot prediction distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=predictions_df, x=f'{model_name}_Prediction', bins=50)
    plt.title(f'{model_name} Prediction Distribution')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Sample Count')
    
    # Plot ROC curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(predictions_df['True_Label'], predictions_df[f'{model_name}_Prediction'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name.lower()}_predictions.png')
    plt.close()

def plot_ensemble_comparison(predictions_df):
    """Plot comparison between ensemble model and individual models"""
    plt.figure(figsize=(15, 10))
    
    # Plot ROC curves for all models
    plt.subplot(2, 2, 1)
    for model in ['RNN', 'DenseNet', 'CNN-LSTM', 'WaveNet', 'Ensemble']:
        fpr, tpr, _ = roc_curve(predictions_df['True_Label'], predictions_df[f'{model}_Prediction'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    
    # Plot prediction boxplots using seaborn instead of pandas
    plt.subplot(2, 2, 2)
    pred_columns = [col for col in predictions_df.columns if col.endswith('_Prediction')]
    plot_data = pd.melt(predictions_df[pred_columns])
    sns.boxplot(x='variable', y='value', data=plot_data)
    plt.title('Model Predictions Distribution')
    plt.ylabel('Prediction Probability')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    
    # Plot confusion matrix
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(predictions_df['True_Label'], 
                         predictions_df['Ensemble_Prediction'] > 0.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot PR curves
    plt.subplot(2, 2, 4)
    for model in ['RNN', 'DenseNet', 'CNN-LSTM', 'WaveNet', 'Ensemble']:
        precision, recall, _ = precision_recall_curve(predictions_df['True_Label'], 
                                                    predictions_df[f'{model}_Prediction'])
        avg_precision = average_precision_score(predictions_df['True_Label'], 
                                             predictions_df[f'{model}_Prediction'])
        plt.plot(recall, precision, label=f'{model} (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig('outputs/ensemble_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 加载预测结果
    predictions_df = pd.read_csv('outputs/predictions.csv')
    
    # 加载训练历史
    try:
        rnn_history = pd.read_csv('RNN/logs/fold_1/training_fold_1.csv')
        densenet_history = pd.read_csv('DenseNet/logs/fold_1/training_fold_1.csv')
        cnn_lstm_history = pd.read_csv('CNN-LSTM/logs/fold_1/training_fold_1.csv')
        wavenet_history = pd.read_csv('outputs/wavenet/logs/wavenet_history.csv')
    except FileNotFoundError as e:
        print(f"警告：无法加载训练历史文件: {str(e)}")
        print("将跳过训练历史相关的可视化。")
        rnn_history = None
        densenet_history = None
        cnn_lstm_history = None
        wavenet_history = None
    
    # 生成可视化
    if all([rnn_history is not None, densenet_history is not None, 
            cnn_lstm_history is not None, wavenet_history is not None]):
        plot_training_history(rnn_history, densenet_history, cnn_lstm_history, wavenet_history, 'loss')
        plot_training_history(rnn_history, densenet_history, cnn_lstm_history, wavenet_history, 'accuracy')
    
    plot_prediction_distribution(predictions_df)
    plot_confusion_matrices(predictions_df)
    create_summary_table(predictions_df)
    plot_ensemble_comparison(predictions_df)
    
    # 为每个模型单独生成预测结果图
    for model in ['RNN', 'DenseNet', 'CNN-LSTM', 'WaveNet']:
        plot_model_predictions(predictions_df, model)

if __name__ == "__main__":
    main()

# 执行命令：
# python visualize_results.py 