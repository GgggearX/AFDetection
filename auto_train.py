#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
顺序执行训练脚本
增强版本 - 执行多个训练文件并保存性能结果
"""

import subprocess
import time
import datetime
import os
import json
import pandas as pd
import sys

# 要执行的训练文件列表及其对应的模型目录
TRAINING_CONFIGS = [
    {"script": "train_wavenet.py", "model_dir": "WaveNet", "model_name": "WaveNet"},
    {"script": "train_densenet.py", "model_dir": "DenseNet", "model_name": "DenseNet"},
    {"script": "train_CNN-LSTM.py", "model_dir": "CNN-LSTM", "model_name": "CNN-LSTM"}, 
    {"script": "train_rnn.py", "model_dir": "RNN", "model_name": "RNN-GRU"},
    {"script": "train_transformer.py", "model_dir": "Transformer", "model_name": "Transformer"}
]

# 要收集的性能指标
PERFORMANCE_METRICS = [
    "val_f1_score", "val_auc_1", "val_accuracy", "val_precision_1", "val_recall_1"
]

# 创建日志目录
if not os.path.exists("training_logs"):
    os.makedirs("training_logs")

# 创建日志文件
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"training_logs/training_run_{timestamp}.log"
results_csv = f"training_logs/performance_summary_{timestamp}.csv"

# 性能结果存储
performance_results = []

def collect_model_performance(model_config):
    """收集模型的性能指标"""
    model_name = model_config["model_name"]
    model_dir = model_config["model_dir"]
    summary_path = os.path.join(model_dir, "results", "cv_summary.json")
    
    if not os.path.exists(summary_path):
        print(f"警告: 未找到{model_name}的性能摘要文件: {summary_path}")
        return {"model": model_name, "training_completed": False}
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # 提取所需的性能指标
        performance = {"model": model_name, "training_completed": True}
        for metric in PERFORMANCE_METRICS:
            if metric in summary:
                performance[metric] = summary[metric]
            else:
                performance[metric] = None
        
        return performance
    except Exception as e:
        print(f"读取{model_name}性能摘要时出错: {str(e)}")
        return {"model": model_name, "training_completed": False, "error": str(e)}

def format_performance_for_log(performance):
    """格式化性能指标用于日志记录"""
    if not performance.get("training_completed", False):
        return "训练未完成或结果文件不存在"
    
    lines = []
    lines.append(f"模型: {performance['model']}")
    for metric in PERFORMANCE_METRICS:
        if metric in performance and performance[metric] is not None:
            # 美化指标名称
            metric_name = metric.replace("val_", "验证").replace("_", " ").title()
            lines.append(f"  {metric_name}: {performance[metric]:.4f}")
    
    return "\n".join(lines)

# 顺序执行训练脚本
print(f"开始执行训练序列 - 日志保存在 {log_file}")

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"=== 训练序列开始: {datetime.datetime.now()} ===\n\n")
    f.write(f"以下脚本将依次执行:\n")
    for i, config in enumerate(TRAINING_CONFIGS):
        f.write(f"{i+1}. {config['script']} (模型: {config['model_name']})\n")
    f.write("\n" + "="*50 + "\n\n")

for i, config in enumerate(TRAINING_CONFIGS):
    script = config["script"]
    model_name = config["model_name"]
    print(f"\n[{i+1}/{len(TRAINING_CONFIGS)}] 开始执行: {script} (模型: {model_name})")
    
    # 记录开始时间
    start_time = time.time()
    
    # 构建命令
    command = f"python {script}"
    
    # 记录到日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] 开始执行: {command}\n")
    
    # 执行命令
    try:
        process = subprocess.run(command, shell=True, check=True, 
                                 stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                                 encoding='utf-8')
        execution_successful = True
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {str(e)}")
        execution_successful = False
        error_output = e.stderr
    
    # 计算经过时间
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
    
    # 记录执行结果
    result = "成功" if execution_successful else "失败"
    result_message = f"[{i+1}/{len(TRAINING_CONFIGS)}] {script} 执行{result}，耗时: {time_str}"
    print(result_message)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {result_message}\n")
        if not execution_successful:
            f.write("错误信息:\n")
            f.write(error_output)
            f.write("\n")

    # 收集性能指标
    performance = collect_model_performance(config)
    performance_results.append(performance)
    
    # 将性能指标写入日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n性能指标:\n")
        f.write(format_performance_for_log(performance))
        f.write("\n\n" + "-"*50 + "\n\n")

# 生成性能汇总表格
try:
    df = pd.DataFrame(performance_results)
    
    # 重命名列以便更易读
    column_mapping = {
        "model": "模型",
        "training_completed": "训练完成",
        "val_f1_score": "F1分数",
        "val_auc_1": "AUC",
        "val_accuracy": "准确率",
        "val_precision_1": "精确率",
        "val_recall_1": "召回率"
    }
    df = df.rename(columns=column_mapping)
    
    # 保存为CSV
    df.to_csv(results_csv, index=False, encoding='utf-8')
    
    # 生成性能汇总表
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n=== 性能汇总 ===\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
except Exception as e:
    print(f"生成性能汇总表格时出错: {str(e)}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n生成性能汇总表格时出错: {str(e)}\n")

# 记录完成时间
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n=== 训练序列完成: {datetime.datetime.now()} ===\n")
    f.write(f"性能汇总已保存至: {results_csv}\n")

print(f"\n所有训练脚本执行完毕。")
print(f"详细日志: {log_file}")
print(f"性能汇总: {results_csv}") 