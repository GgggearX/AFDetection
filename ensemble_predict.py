import numpy as np
import pandas as pd
import os
import h5py
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from datasets import ECGSequence, ECGPredictSequence

def load_best_models(rnn_dir='RNN', densenet_dir='DenseNet', cnn_lstm_dir='CNN-LSTM', wavenet_dir='outputs/wavenet'):
    """加载每个fold的最佳模型"""
    rnn_models = []
    densenet_models = []
    cnn_lstm_models = []
    wavenet_models = []
    
    # 加载RNN模型
    print("\n加载RNN模型:")
    for fold in range(1, 6):
        model_path = os.path.join(rnn_dir, 'models', f'fold_{fold}', f'model_best_rnn_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"找到RNN模型 fold {fold}: {model_path}")
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
                # 验证模型输出
                input_shape = model.input_shape[1:]
                test_input = np.random.random((1,) + input_shape)
                test_output = model.predict(test_input, verbose=0)
                if test_output.shape == (1, 1):
                    rnn_models.append(model)
                    print(f"成功验证RNN模型 fold {fold}")
                else:
                    print(f"警告：RNN模型 fold {fold} 输出维度不正确: {test_output.shape}")
            except Exception as e:
                print(f"加载RNN模型 fold {fold} 时出错: {str(e)}")
        else:
            print(f"警告：未找到RNN模型 fold {fold}，路径: {model_path}")
    
    # 加载DenseNet模型
    print("\n加载DenseNet模型:")
    for fold in range(1, 6):
        model_path = os.path.join(densenet_dir, 'models', f'fold_{fold}', f'model_best_densenet_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"找到DenseNet模型 fold {fold}: {model_path}")
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
                input_shape = model.input_shape[1:]
                test_input = np.random.random((1,) + input_shape)
                test_output = model.predict(test_input, verbose=0)
                if test_output.shape == (1, 1):
                    densenet_models.append(model)
                    print(f"成功验证DenseNet模型 fold {fold}")
                else:
                    print(f"警告：DenseNet模型 fold {fold} 输出维度不正确: {test_output.shape}")
            except Exception as e:
                print(f"加载DenseNet模型 fold {fold} 时出错: {str(e)}")
        else:
            print(f"警告：未找到DenseNet模型 fold {fold}，路径: {model_path}")
    
    # 加载CNN-LSTM模型
    print("\n加载CNN-LSTM模型:")
    for fold in range(1, 6):
        model_path = os.path.join(cnn_lstm_dir, 'models', f'fold_{fold}', f'model_best_cnn_lstm_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"找到CNN-LSTM模型 fold {fold}: {model_path}")
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
                input_shape = model.input_shape[1:]
                test_input = np.random.random((1,) + input_shape)
                test_output = model.predict(test_input, verbose=0)
                if test_output.shape == (1, 1):
                    cnn_lstm_models.append(model)
                    print(f"成功验证CNN-LSTM模型 fold {fold}")
                else:
                    print(f"警告：CNN-LSTM模型 fold {fold} 输出维度不正确: {test_output.shape}")
            except Exception as e:
                print(f"加载CNN-LSTM模型 fold {fold} 时出错: {str(e)}")
        else:
            print(f"警告：未找到CNN-LSTM模型 fold {fold}，路径: {model_path}")
    
    # 加载WaveNet模型
    print("\n加载WaveNet模型:")
    model_path = os.path.join(wavenet_dir, 'checkpoints', 'wavenet_model_best.h5')
    if os.path.exists(model_path):
        print(f"找到WaveNet模型: {model_path}")
        try:
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
            input_shape = model.input_shape[1:]
            test_input = np.random.random((1,) + input_shape)
            test_output = model.predict(test_input, verbose=0)
            if test_output.shape == (1, 1):
                wavenet_models.append(model)
                print("成功验证WaveNet模型")
            else:
                print(f"警告：WaveNet模型输出维度不正确: {test_output.shape}")
        except Exception as e:
            print(f"加载WaveNet模型时出错: {str(e)}")
    else:
        print(f"警告：未找到WaveNet模型，路径: {model_path}")
    
    if len(rnn_models) != 5 or len(densenet_models) != 5 or len(cnn_lstm_models) != 5 or len(wavenet_models) != 1:
        print(f"\n警告：期望每个类型有指定数量的模型，但实际找到：")
        print(f"RNN模型: {len(rnn_models)}")
        print(f"DenseNet模型: {len(densenet_models)}")
        print(f"CNN-LSTM模型: {len(cnn_lstm_models)}")
        print(f"WaveNet模型: {len(wavenet_models)}")
        print("请确保所有模型都已正确训练和保存。")
    
    print(f"\n总结：已加载 {len(rnn_models)} 个RNN模型, {len(densenet_models)} 个DenseNet模型, {len(cnn_lstm_models)} 个CNN-LSTM模型, {len(wavenet_models)} 个WaveNet模型")
    return rnn_models, densenet_models, cnn_lstm_models, wavenet_models

def predict_with_models(models, x_data, batch_size=32, model_type="Unknown", seq_length=4096, model_dir=None):
    """使用多个模型进行预测并平均结果"""
    if not models:
        print(f"Warning: No {model_type} models available for prediction")
        return None
    
    print(f"\nMaking predictions with {model_type} models:")
    predictions = []
    
    # 根据模型类型准备数据
    x_data_model = x_data[:, :seq_length, :] if seq_length else x_data
    
    for i, model in enumerate(models):
        print(f"Running prediction with {model_type} model {i+1}/{len(models)}")
        # 加载对应的标准化参数
        scaler_path = os.path.join(model_dir, 'models', f'fold_{i+1}', 'scalers.npy') if model_type != 'WaveNet' else os.path.join(model_dir, 'scalers.npy')
        if os.path.exists(scaler_path):
            print(f"Loading scalers from: {scaler_path}")
            scalers = ECGPredictSequence.load_scalers(scaler_path)
            # 创建数据生成器，使用加载的标准化器
            data_sequence = ECGPredictSequence(x_data_model, batch_size=batch_size, scalers=scalers)
            # 使用model.predict进行预测
            pred = model.predict(data_sequence, verbose=1)
            print(f"Prediction shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
            predictions.append(pred)
        else:
            print(f"Warning: Scaler parameters not found at {scaler_path}")
            return None
    
    mean_pred = np.mean(predictions, axis=0)
    print(f"Mean {model_type} prediction shape: {mean_pred.shape}, range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    return mean_pred

def ensemble_predict(x_data, rnn_models, densenet_models, cnn_lstm_models, wavenet_models, batch_size=32):
    """集成RNN、DenseNet、CNN-LSTM和WaveNet的预测结果"""
    print("\n准备预测...")
    
    # 获取四种模型的预测
    print("\nRNN预测 (序列长度: 4096)...")
    rnn_pred = predict_with_models(rnn_models, x_data, batch_size, "RNN", seq_length=4096, model_dir='RNN')
    
    print("\nDenseNet预测 (序列长度: 4096)...")
    densenet_pred = predict_with_models(densenet_models, x_data, batch_size, "DenseNet", seq_length=4096, model_dir='DenseNet')
    
    print("\nCNN-LSTM预测 (序列长度: 4096)...")
    cnn_lstm_pred = predict_with_models(cnn_lstm_models, x_data, batch_size, "CNN-LSTM", seq_length=4096, model_dir='CNN-LSTM')
    
    print("\nWaveNet预测 (序列长度: 4096)...")
    wavenet_pred = predict_with_models(wavenet_models, x_data, batch_size, "WaveNet", seq_length=4096, model_dir='outputs/wavenet')
    
    if rnn_pred is None or densenet_pred is None or cnn_lstm_pred is None or wavenet_pred is None:
        print("错误：一个或多个模型预测失败")
        return None, rnn_pred, densenet_pred, cnn_lstm_pred, wavenet_pred
    
    # 加权平均
    print("\n计算集成预测结果...")
    weights = [0.2, 0.2, 0.3, 0.3]  # 给CNN-LSTM和WaveNet稍微高一点的权重
    ensemble_pred = (weights[0] * rnn_pred + weights[1] * densenet_pred + 
                    weights[2] * cnn_lstm_pred + weights[3] * wavenet_pred)
    print(f"集成预测形状: {ensemble_pred.shape}, 范围: [{ensemble_pred.min():.3f}, {ensemble_pred.max():.3f}]")
    
    return ensemble_pred, rnn_pred, densenet_pred, cnn_lstm_pred, wavenet_pred

def plot_roc_curves(y_true, rnn_pred, densenet_pred, cnn_lstm_pred, wavenet_pred, ensemble_pred):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    # 计算每个模型的ROC曲线
    for pred, label, color in [(rnn_pred, 'RNN', 'blue'),
                              (densenet_pred, 'DenseNet', 'red'),
                              (cnn_lstm_pred, 'CNN-LSTM', 'orange'),
                              (wavenet_pred, 'WaveNet', 'purple'),
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

def plot_precision_recall_curves(y_true, rnn_pred, densenet_pred, cnn_lstm_pred, wavenet_pred, ensemble_pred):
    """绘制PR曲线"""
    plt.figure(figsize=(10, 8))
    
    # 计算每个模型的PR曲线
    for pred, label, color in [(rnn_pred, 'RNN', 'blue'),
                              (densenet_pred, 'DenseNet', 'red'),
                              (cnn_lstm_pred, 'CNN-LSTM', 'orange'),
                              (wavenet_pred, 'WaveNet', 'purple'),
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
    
    # 设置默认路径
    default_data_dir = 'data/physionet/training2017'
    default_reference_file = 'data/REFERENCE-v3.csv'
    
    # 加载数据
    print("加载数据...")
    from datasets import load_data
    x_data, y_true = load_data(default_data_dir, default_reference_file, max_seq_length=4096)
    
    # 加载模型
    print("加载模型...")
    rnn_models, densenet_models, cnn_lstm_models, wavenet_models = load_best_models()
    
    if len(rnn_models) == 0 or len(densenet_models) == 0 or len(cnn_lstm_models) == 0 or len(wavenet_models) == 0:
        print("错误：未找到模型。请确保所有模型都已正确训练和保存。")
        return
    
    # 进行预测
    print("开始预测...")
    ensemble_pred, rnn_pred, densenet_pred, cnn_lstm_pred, wavenet_pred = ensemble_predict(
        x_data, rnn_models, densenet_models, cnn_lstm_models, wavenet_models)
    
    if ensemble_pred is None:
        print("错误：预测失败。请检查模型和数据格式。")
        return
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'RNN_Prediction': rnn_pred.flatten(),
        'DenseNet_Prediction': densenet_pred.flatten(),
        'CNN-LSTM_Prediction': cnn_lstm_pred.flatten(),
        'WaveNet_Prediction': wavenet_pred.flatten(),
        'Ensemble_Prediction': ensemble_pred.flatten(),
        'True_Label': y_true.flatten()
    })
    results_df.to_csv('outputs/predictions.csv', index=False)
    
    # 绘制评估图表
    print("生成评估图表...")
    plot_roc_curves(y_true.flatten(), rnn_pred.flatten(), 
                   densenet_pred.flatten(), cnn_lstm_pred.flatten(),
                   wavenet_pred.flatten(), ensemble_pred.flatten())
    plot_precision_recall_curves(y_true.flatten(), rnn_pred.flatten(), 
                               densenet_pred.flatten(), cnn_lstm_pred.flatten(),
                               wavenet_pred.flatten(), ensemble_pred.flatten())
    
    # 打印评估指标
    print("\n模型评估指标:")
    for model in ['RNN', 'DenseNet', 'CNN-LSTM', 'WaveNet', 'Ensemble']:
        pred = results_df[f'{model}_Prediction']
        true = results_df['True_Label']
        
        # 计算指标
        auc_score = roc_auc_score(true, pred)
        ap_score = average_precision_score(true, pred)
        f1 = f1_score(true, pred > 0.5)
        precision = precision_score(true, pred > 0.5)
        recall = recall_score(true, pred > 0.5)
        
        print(f"\n{model} 模型:")
        print(f"AUC: {auc_score:.3f}")
        print(f"Average Precision: {ap_score:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")

if __name__ == "__main__":
    main()

# 执行命令：
# python ensemble_predict.py 