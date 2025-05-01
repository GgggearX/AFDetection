# ECG原始数据集分布分析报告
## 1. 原始数据特性
![longest_records.png](raw_data\longest_records.png)
![record_length_distribution.png](raw_data\record_length_distribution.png)
![sampling_rate_distribution.png](raw_data\sampling_rate_distribution.png)
![shortest_records.png](raw_data\shortest_records.png)

### 数据摘要
- 样本数量: 8528
- 导联数量: 1
- 数据形状: 异构数组，每个样本可能有不同形状

#### 采样率统计
- 最小采样率: 300.0 Hz
- 最大采样率: 300.0 Hz
- 平均采样率: 300.0 Hz

采样率分布:
- 300 Hz: 8528 条记录

#### 记录长度统计
- 最短记录: 2714 样本点
- 最长记录: 18286 样本点
- 平均长度: 9749.240267354597 样本点
- 中位数长度: 9000.0 样本点

## 2. 类别分布
![class_bar_chart.png](class_distribution\class_bar_chart.png)
![class_pie_chart.png](class_distribution\class_pie_chart.png)

## 3. 信号特征分布
![signal_mean_distribution.png](signal_features\signal_mean_distribution.png)
![signal_range_distribution.png](signal_features\signal_range_distribution.png)
![signal_std_distribution.png](signal_features\signal_std_distribution.png)
![信号均值_boxplot.png](signal_features\信号均值_boxplot.png)
![信号幅值范围_boxplot.png](signal_features\信号幅值范围_boxplot.png)
![信号标准差_boxplot.png](signal_features\信号标准差_boxplot.png)

## 4. 时域信号示例
![af_signal_example_1.png](time_domain\af_signal_example_1.png)
![af_signal_example_2.png](time_domain\af_signal_example_2.png)
![af_signal_example_3.png](time_domain\af_signal_example_3.png)
![af_signal_example_4.png](time_domain\af_signal_example_4.png)
![af_signal_example_5.png](time_domain\af_signal_example_5.png)
![af_vs_non_af_comparison.png](time_domain\af_vs_non_af_comparison.png)
![non_af_signal_example_1.png](time_domain\non_af_signal_example_1.png)
![non_af_signal_example_2.png](time_domain\non_af_signal_example_2.png)
![non_af_signal_example_3.png](time_domain\non_af_signal_example_3.png)
![non_af_signal_example_4.png](time_domain\non_af_signal_example_4.png)
![non_af_signal_example_5.png](time_domain\non_af_signal_example_5.png)

## 5. 频域分析
![af_spectrum_1.png](frequency_domain\af_spectrum_1.png)
![af_spectrum_2.png](frequency_domain\af_spectrum_2.png)
![af_spectrum_3.png](frequency_domain\af_spectrum_3.png)
![af_vs_non_af_spectrum_comparison.png](frequency_domain\af_vs_non_af_spectrum_comparison.png)
![non_af_spectrum_1.png](frequency_domain\non_af_spectrum_1.png)
![non_af_spectrum_2.png](frequency_domain\non_af_spectrum_2.png)
![non_af_spectrum_3.png](frequency_domain\non_af_spectrum_3.png)
