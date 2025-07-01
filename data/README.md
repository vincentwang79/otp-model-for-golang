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