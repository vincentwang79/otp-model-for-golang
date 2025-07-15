# OTP 短信检测数据集

## 数据来源

本项目使用了两个数据集：

1. **SMS_OTP_10000_samples.csv** - 包含10,000条OTP短信样本
2. **SMSSpamCollection** - 包含5,574条标记为ham（普通短信）或spam（垃圾短信）的短信样本

其中，SMSSpamCollection数据集的详细说明可在 `data/raw/readme` 文件中找到。

## 数据处理

数据处理流程如下：

1. 从SMS_OTP_10000_samples.csv中提取短信文本，并标记为OTP（is_otp=1）
2. 从SMSSpamCollection中提取短信文本，并标记为非OTP（is_otp=0）
3. 对两个数据集分别进行训练/验证集划分（90%训练，10%验证）
4. 合并数据集并随机打乱

## 数据平衡策略

为解决数据集中OTP和非OTP短信比例不均衡的问题（约64.2%:35.8%），我们实现了两种数据平衡策略：

1. **SMOTE过采样**：生成合成的少数类样本，将少数类样本数量提升到接近多数类水平
2. **随机欠采样**：随机移除多数类样本，使两个类别数量接近

这些平衡策略与增强特征工程相结合，显著提高了模型性能。

## 目录结构

```
data/
├── raw/                  # 原始数据说明
│   └── readme            # SMSSpamCollection数据集说明
├── training-data/        # 原始训练数据
│   ├── SMS_OTP_10000_samples.csv   # OTP短信样本
│   └── SMSSpamCollection           # 非OTP短信样本
└── processed/            # 处理后的数据
    ├── combined_sms_dataset.csv    # 合并后的数据集
    └── dataset_stats.md            # 数据集统计信息
```

## 数据集格式

处理后的数据集（`data/processed/combined_sms_dataset.csv`）包含以下字段：

- **message**: 短信文本内容
- **is_otp**: 是否为OTP短信（1=是，0=否）
- **split**: 数据集划分（'train'=训练集，'val'=验证集）

## 使用方法

可以使用以下脚本处理数据：

```bash
python3 scripts/process_datasets.py
```

处理后的数据集将保存在 `data/processed/combined_sms_dataset.csv`，统计信息将保存在 `data/processed/dataset_stats.md`。 

## 模型性能

基于此数据集训练的平衡增强模型性能如下（按F1分数降序排列）：

| 模型 | 平衡方法 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|---------|-------|-------|-------|-------|-----|
| SVM  | balanced_enhanced | 0.90 | 0.86 | 0.83 | 0.8539 | 0.89 |
| NB   | balanced_enhanced | 0.90 | 0.88 | 0.81 | 0.8433 | 0.91 |
| RF   | balanced_enhanced | 0.88 | 0.81 | 0.84 | 0.8421 | 0.90 |
| NB   | smote | 0.9014 | 0.9140 | 0.7735 | 0.8379 | 0.9139 |
| SVM  | undersample | 0.8002 | 0.6412 | 0.8934 | 0.7465 | 0.8853 |
| SVM  | smote | 0.7643 | 0.6072 | 0.8049 | 0.6922 | 0.8420 |
| RF   | smote | 0.7785 | 0.9053 | 0.3657 | 0.5210 | 0.9183 | 