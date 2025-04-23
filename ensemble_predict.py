import numpy as np
import pandas as pd
import os
import h5py
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from datasets import ECGSequence, ECGPredictSequence
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    """实现 Focal Loss 损失函数"""
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        pt_1 = tf.where(tf.equal(y_true, 1.0), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0.0), y_pred, tf.zeros_like(y_pred))
        loss_1 = -alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1 + epsilon)
        loss_0 = -(1.0 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0 + epsilon)
        return tf.reduce_mean(loss_1 + loss_0)
    return focal_loss_fixed

def f1_score(y_true, y_pred):
    """计算F1分数，使用0.1作为阈值"""
    y_pred_binary = K.cast(K.greater(y_pred, 0.1), K.floatx())
    true_positives = K.sum(y_true * y_pred_binary)
    possible_positives = K.sum(y_true)
    predicted_positives = K.sum(y_pred_binary)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

def load_best_models(rnn_dir='RNN', densenet_dir='DenseNet', cnn_lstm_dir='CNN-LSTM', wavenet_dir='WaveNet', transformer_dir='Transformer'):
    """加载每个fold的最佳模型"""
    rnn_models = []
    densenet_models = []
    cnn_lstm_models = []
    wavenet_models = []
    transformer_models = []
    
    # 修改模型加载部分
    custom_objects = {
        'focal_loss_fixed': focal_loss(),
        'f1_score': f1_score
    }
    
    # 加载RNN模型
    print("\n加载RNN模型:")
    for fold in range(1, 6):
        model_path = os.path.join(rnn_dir, 'models', f'fold_{fold}', f'model_best_rnn_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"找到RNN模型 fold {fold}: {model_path}")
            try:
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                model.compile(
                    optimizer='adam',
                    loss=focal_loss(gamma=2.0, alpha=0.25),
                    metrics=['accuracy', 
                            AUC(name='auc_1'), 
                            Precision(name='precision_1', thresholds=0.1),
                            Recall(name='recall_1', thresholds=0.1),
                            f1_score]
                )
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
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                model.compile(
                    optimizer='adam',
                    loss=focal_loss(gamma=2.0, alpha=0.25),
                    metrics=['accuracy', 
                            AUC(name='auc_1'), 
                            Precision(name='precision_1', thresholds=0.1),
                            Recall(name='recall_1', thresholds=0.1),
                            f1_score]
                )
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
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                model.compile(
                    optimizer='adam',
                    loss=focal_loss(gamma=2.0, alpha=0.25),
                    metrics=['accuracy', 
                            AUC(name='auc_1'), 
                            Precision(name='precision_1', thresholds=0.1),
                            Recall(name='recall_1', thresholds=0.1),
                            f1_score]
                )
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
    for fold in range(1, 6):
        model_path = os.path.join(wavenet_dir, 'models', f'fold_{fold}', f'model_best_wavenet_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"找到WaveNet模型 fold {fold}: {model_path}")
            try:
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                model.compile(
                    optimizer='adam',
                    loss=focal_loss(gamma=2.0, alpha=0.25),
                    metrics=['accuracy', 
                            AUC(name='auc_1'), 
                            Precision(name='precision_1', thresholds=0.1),
                            Recall(name='recall_1', thresholds=0.1),
                            f1_score]
                )
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
    
    # 加载Transformer模型
    print("\n加载Transformer模型:")
    for fold in range(1, 6):
        model_path = os.path.join(transformer_dir, 'models', f'fold_{fold}', f'model_best_transformer_fold_{fold}.hdf5')
        if os.path.exists(model_path):
            print(f"找到Transformer模型 fold {fold}: {model_path}")
            try:
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                model.compile(
                    optimizer='adam',
                    loss=focal_loss(gamma=2.0, alpha=0.25),
                    metrics=['accuracy', 
                            AUC(name='auc_1'), 
                            Precision(name='precision_1', thresholds=0.1),
                            Recall(name='recall_1', thresholds=0.1),
                            f1_score]
                )
                input_shape = model.input_shape[1:]
                test_input = np.random.random((1,) + input_shape)
                test_output = model.predict(test_input, verbose=0)
                if test_output.shape == (1, 1):
                    transformer_models.append(model)
                    print(f"成功验证Transformer模型 fold {fold}")
                else:
                    print(f"警告：Transformer模型 fold {fold} 输出维度不正确: {test_output.shape}")
            except Exception as e:
                print(f"加载Transformer模型 fold {fold} 时出错: {str(e)}")
        else:
            print(f"警告：未找到Transformer模型 fold {fold}，路径: {model_path}")
    
    if len(rnn_models) != 5 or len(densenet_models) != 5 or len(cnn_lstm_models) != 5 or len(wavenet_models) != 1 or len(transformer_models) != 5:
        print(f"\n警告：期望每个类型有指定数量的模型，但实际找到：")
        print(f"RNN模型: {len(rnn_models)}")
        print(f"DenseNet模型: {len(densenet_models)}")
        print(f"CNN-LSTM模型: {len(cnn_lstm_models)}")
        print(f"WaveNet模型: {len(wavenet_models)}")
        print(f"Transformer模型: {len(transformer_models)}")
        print("请确保所有模型都已正确训练和保存。")
    
    print(f"\n总结：已加载 {len(rnn_models)} 个RNN模型, {len(densenet_models)} 个DenseNet模型, {len(cnn_lstm_models)} 个CNN-LSTM模型, {len(wavenet_models)} 个WaveNet模型, {len(transformer_models)} 个Transformer模型")
    return rnn_models, densenet_models, cnn_lstm_models, wavenet_models, transformer_models

def predict_with_models(models, x_data, batch_size, model_name, seq_length=4096, model_dir=None):
    """使用指定模型进行预测"""
    if not models:
        print(f"警告: 没有找到可用的{model_name}模型")
        return None
    
    predictions = []
    for i, model in enumerate(models, 1):
        print(f"\n使用{model_name} fold {i}进行预测...")
        if model_dir:
            data_generator = ECGPredictSequence(x_data, batch_size=batch_size, max_seq_length=seq_length)
        else:
            data_generator = ECGPredictSequence(x_data, batch_size=batch_size, max_seq_length=seq_length)
        pred = model.predict(data_generator, verbose=1)
        predictions.append(pred)
    
    return np.mean(predictions, axis=0) if predictions else None

def evaluate_predictions(y_true, y_pred, model_name):
    """评估预测结果"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # 计算F1分数（使用0.1作为阈值）
    y_pred_binary = (y_pred > 0.1).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    print(f"\n{model_name}模型评估结果:")
    print(f"AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return roc_auc, f1, precision, recall

def main():
    """主函数"""
    # 加载数据
    data_dir = "data/physionet/training2017"
    reference_file = "data/physionet/training2017/REFERENCE.csv"
    
    # 加载标签
    labels_df = pd.read_csv(reference_file, header=None, names=['file', 'label'])
    labels_df['label'] = (labels_df['label'] == 'A').astype(int)
    
    # 加载所有模型
    rnn_models, densenet_models, cnn_lstm_models, wavenet_models, transformer_models = load_best_models()
    
    # 设置批量大小
    batch_size = 32
    
    # 加载测试数据
    x_data = []
    y_true = []
    for idx, row in labels_df.iterrows():
        file_path = os.path.join(data_dir, row['file'] + '.mat')
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                data = np.array(f['val'])
                x_data.append(data)
                y_true.append(row['label'])
    
    x_data = np.array(x_data)
    y_true = np.array(y_true)
    
    # 进行预测
    print("\nRNN预测 (序列长度: 4096)...")
    rnn_pred = predict_with_models(rnn_models, x_data, batch_size, "RNN", seq_length=4096)
    
    print("\nDenseNet预测 (序列长度: 4096)...")
    densenet_pred = predict_with_models(densenet_models, x_data, batch_size, "DenseNet", seq_length=4096)
    
    print("\nCNN-LSTM预测 (序列长度: 4096)...")
    cnn_lstm_pred = predict_with_models(cnn_lstm_models, x_data, batch_size, "CNN-LSTM", seq_length=4096)
    
    print("\nWaveNet预测 (序列长度: 4096)...")
    wavenet_pred = predict_with_models(wavenet_models, x_data, batch_size, "WaveNet", seq_length=4096)
    
    print("\nTransformer预测 (序列长度: 4096)...")
    transformer_pred = predict_with_models(transformer_models, x_data, batch_size, "Transformer", seq_length=4096)
    
    # 评估每个模型的性能
    results = {}
    if rnn_pred is not None:
        results['RNN'] = evaluate_predictions(y_true, rnn_pred, "RNN")
    if densenet_pred is not None:
        results['DenseNet'] = evaluate_predictions(y_true, densenet_pred, "DenseNet")
    if cnn_lstm_pred is not None:
        results['CNN-LSTM'] = evaluate_predictions(y_true, cnn_lstm_pred, "CNN-LSTM")
    if wavenet_pred is not None:
        results['WaveNet'] = evaluate_predictions(y_true, wavenet_pred, "WaveNet")
    if transformer_pred is not None:
        results['Transformer'] = evaluate_predictions(y_true, transformer_pred, "Transformer")
    
    # 集成预测（简单平均）
    valid_predictions = [pred for pred in [rnn_pred, densenet_pred, cnn_lstm_pred, wavenet_pred, transformer_pred] if pred is not None]
    if valid_predictions:
        ensemble_pred = np.mean(valid_predictions, axis=0)
        print("\n集成模型评估结果:")
        ensemble_results = evaluate_predictions(y_true, ensemble_pred, "Ensemble")
        results['Ensemble'] = ensemble_results
    
    # 保存结果
    results_df = pd.DataFrame(results, index=['AUC', 'F1', 'Precision', 'Recall']).T
    results_df.to_csv('ensemble_results.csv')
    print("\n结果已保存到 ensemble_results.csv")

if __name__ == "__main__":
    main()

# 执行命令：
# python ensemble_predict.py 