import numpy as np
import pandas as pd
import os
import h5py
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from datasets import ECGSequence

def load_best_models(rnn_dir='RNN', densenet_dir='DenseNet'):
    """加载每个fold的最佳模型"""
    rnn_models = []
    densenet_models = []
    
    # 加载RNN模型
    print("\nLoading RNN models:")
    for fold in range(1, 6):
        model_path = os.path.join(rnn_dir, 'models', f'fold_{fold}', f'model_best_rnn_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"Found RNN model for fold {fold}: {model_path}")
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
                # 验证模型输出
                test_input = np.random.random((1, 4096, 12))
                test_output = model.predict(test_input, verbose=0)
                if test_output.shape == (1, 1):
                    rnn_models.append(model)
                    print(f"Successfully validated RNN model for fold {fold}")
                else:
                    print(f"Warning: RNN model for fold {fold} has incorrect output shape: {test_output.shape}")
            except Exception as e:
                print(f"Error loading RNN model for fold {fold}: {str(e)}")
        else:
            print(f"Warning: RNN model for fold {fold} not found at {model_path}")
    
    # 加载DenseNet模型
    print("\nLoading DenseNet models:")
    for fold in range(1, 6):
        model_path = os.path.join(densenet_dir, 'models', f'fold_{fold}', f'model_best_densenet_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"Found DenseNet model for fold {fold}: {model_path}")
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
                # 验证模型输出
                test_input = np.random.random((1, 4096, 12))
                test_output = model.predict(test_input, verbose=0)
                if test_output.shape == (1, 1):
                    densenet_models.append(model)
                    print(f"Successfully validated DenseNet model for fold {fold}")
                else:
                    print(f"Warning: DenseNet model for fold {fold} has incorrect output shape: {test_output.shape}")
            except Exception as e:
                print(f"Error loading DenseNet model for fold {fold}: {str(e)}")
        else:
            print(f"Warning: DenseNet model for fold {fold} not found at {model_path}")
    
    if len(rnn_models) != 5 or len(densenet_models) != 5:
        print(f"\nWarning: Expected 5 models for each type, but found:")
        print(f"RNN models: {len(rnn_models)}")
        print(f"DenseNet models: {len(densenet_models)}")
        print("Please ensure all models are trained and saved correctly.")
    
    print(f"\nSummary: Loaded {len(rnn_models)} RNN models and {len(densenet_models)} DenseNet models")
    return rnn_models, densenet_models

def predict_with_models(models, data_sequence, model_type="Unknown"):
    """使用多个模型进行预测并平均结果"""
    if not models:
        print(f"Warning: No {model_type} models available for prediction")
        return None
    
    print(f"\nMaking predictions with {model_type} models:")
    predictions = []
    for i, model in enumerate(models):
        print(f"Running prediction with {model_type} model {i+1}/{len(models)}")
        # 使用model.predict进行预测
        pred = model.predict(data_sequence, verbose=1)
        print(f"Prediction shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
        predictions.append(pred)
    
    mean_pred = np.mean(predictions, axis=0)
    print(f"Mean {model_type} prediction shape: {mean_pred.shape}, range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    return mean_pred

def ensemble_predict(x_data, rnn_models, densenet_models, batch_size=32):
    """集成RNN和DenseNet的预测结果"""
    print("\nPreparing data for prediction...")
    data_sequence = ECGSequence(x_data, batch_size=batch_size)
    
    # 获取两种模型的预测
    rnn_pred = predict_with_models(rnn_models, data_sequence, "RNN")
    densenet_pred = predict_with_models(densenet_models, data_sequence, "DenseNet")
    
    if rnn_pred is None or densenet_pred is None:
        print("Error: One or both model types failed to produce predictions")
        return None, rnn_pred, densenet_pred
    
    # 加权平均（可以调整权重）
    print("\nComputing ensemble predictions...")
    ensemble_pred = 0.5 * rnn_pred + 0.5 * densenet_pred
    print(f"Ensemble prediction shape: {ensemble_pred.shape}, range: [{ensemble_pred.min():.3f}, {ensemble_pred.max():.3f}]")
    
    return ensemble_pred, rnn_pred, densenet_pred

def plot_roc_curves(y_true, rnn_pred, densenet_pred, ensemble_pred):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    # 计算每个模型的ROC曲线
    for pred, label, color in [(rnn_pred, 'RNN', 'blue'),
                              (densenet_pred, 'DenseNet', 'red'),
                              (ensemble_pred, 'Ensemble', 'green')]:
        fpr, tpr, _ = roc_curve(y_true, pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f'{label} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('outputs/roc_curves.png')
    plt.close()

def plot_precision_recall_curves(y_true, rnn_pred, densenet_pred, ensemble_pred):
    """绘制PR曲线"""
    plt.figure(figsize=(10, 8))
    
    # 计算每个模型的PR曲线
    for pred, label, color in [(rnn_pred, 'RNN', 'blue'),
                              (densenet_pred, 'DenseNet', 'red'),
                              (ensemble_pred, 'Ensemble', 'green')]:
        precision, recall, _ = precision_recall_curve(y_true, pred)
        avg_precision = average_precision_score(y_true, pred)
        plt.plot(recall, precision, color=color,
                label=f'{label} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig('outputs/pr_curves.png')
    plt.close()

def main():
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 加载数据
    with h5py.File('data/ecg_tracings.hdf5', 'r') as f:
        x_data = f['tracings'][:]
    
    # 加载标签
    labels_df = pd.read_csv('data/annotations/gold_standard.csv')
    y_true = labels_df['AF'].values
    
    # 加载模型
    print("Loading models...")
    rnn_models, densenet_models = load_best_models()
    
    if len(rnn_models) == 0 or len(densenet_models) == 0:
        print("Error: No models found. Please ensure all models are trained and saved correctly.")
        return
    
    # 进行预测
    print("Making predictions...")
    ensemble_pred, rnn_pred, densenet_pred = ensemble_predict(
        x_data, rnn_models, densenet_models)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'RNN_Prediction': rnn_pred.flatten(),
        'DenseNet_Prediction': densenet_pred.flatten(),
        'Ensemble_Prediction': ensemble_pred.flatten(),
        'True_Label': y_true
    })
    results_df.to_csv('outputs/predictions.csv', index=False)
    
    # 绘制评估图表
    print("Generating evaluation plots...")
    plot_roc_curves(y_true, rnn_pred.flatten(), 
                   densenet_pred.flatten(), ensemble_pred.flatten())
    plot_precision_recall_curves(y_true, rnn_pred.flatten(), 
                               densenet_pred.flatten(), ensemble_pred.flatten())
    
    # 打印评估指标
    print("\n模型评估结果:")
    for name, pred in [('RNN', rnn_pred), ('DenseNet', densenet_pred), 
                      ('Ensemble', ensemble_pred)]:
        ap_score = average_precision_score(y_true, pred)
        print(f"{name} Average Precision: {ap_score:.3f}")

if __name__ == "__main__":
    main()

# 执行命令：
# python ensemble_predict.py 