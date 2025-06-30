# OTP检测器 - Python训练，Go使用

这个项目实现了一个高性能的OTP（一次性密码）短信检测器，使用Python训练模型并在Go中使用。

## 项目结构

```
otp-py-train-go-use/
├── data/                  # 训练和测试数据
│   ├── otp_messages_1000.txt     # OTP短信样本
│   └── non_otp_messages_1000.txt # 非OTP短信样本
├── models/                # 模型文件
│   └── go_params/         # 导出的模型参数
│       ├── otp_svm_params.json   # 模型参数JSON
│       ├── otp_svm.joblib        # 原始Python模型
│       └── processed_examples.txt # 处理后的示例文本
├── src/                   # 源代码
│   ├── go/                # Go实现
│   │   └── detector/      # OTP检测器
│   │       └── detector.go # 检测器实现
│   └── python/            # Python实现
│       └── train_export_model.py # 训练和导出模型
└── test/                  # 测试程序
    ├── benchmark/         # 基准测试
    │   └── main.go        # 基准测试程序
    └── cli/               # 命令行工具
        └── main.go        # CLI程序
```

## 使用方法

### 1. 训练和导出模型

```bash
# 创建Python虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install scikit-learn pandas numpy joblib

# 训练和导出模型
cd src/python
python train_export_model.py
```

### 2. 在Go中使用模型

```bash
# 运行CLI工具
cd test/cli
go run main.go

# 交互模式
go run main.go -interactive

# 基准测试
go run main.go -benchmark

# 处理文件
go run main.go -file path/to/messages.txt
```

## 实现细节

### Python部分

- 使用`LinearSVC`训练线性SVM模型
- 使用`CountVectorizer`进行文本向量化
- 提取文本特征，包括词频、数字模式和关键词
- 导出模型参数为JSON格式

### Go部分

- 实现相同的预处理和特征提取逻辑
- 使用导出的参数进行推理
- 高性能实现，每秒处理约10万条消息

## 性能

- **准确率**：
  - OTP消息：100%
  - 非OTP消息：100%
  - 总体准确率：100%
  - F1分数：1.0

- **性能**：
  - 处理速度：每秒约10万条消息
  - 每条消息平均处理时间：约10微秒

## 许可证

MIT 

# OTP 短信检测数据处理

本项目包含用于清洗和准备OTP（一次性密码）短信检测数据集的代码和数据。

## 项目结构

```
otp_model_for_golang/
├── data/                  # 数据目录
│   ├── raw/               # 原始数据说明
│   ├── training-data/     # 原始训练数据
│   └── processed/         # 处理后的数据
├── scripts/               # 数据处理脚本
│   └── process_datasets.py # 数据清洗和合并脚本
└── README.md              # 项目说明
```

## 数据集

本项目使用了两个数据集：

1. **SMS_OTP_10000_samples.csv** - 包含10,000条OTP短信样本
2. **SMSSpamCollection** - 包含5,574条标记为ham（普通短信）或spam（垃圾短信）的短信样本

处理后的数据集包含以下字段：
- **message**: 短信文本内容
- **is_otp**: 是否为OTP短信（1=是，0=否）
- **split**: 数据集划分（'train'=训练集，'val'=验证集）

## 数据处理流程

数据处理流程如下：

1. 从SMS_OTP_10000_samples.csv中提取短信文本，并标记为OTP（is_otp=1）
2. 从SMSSpamCollection中提取短信文本，并标记为非OTP（is_otp=0）
3. 对两个数据集分别进行训练/验证集划分（90%训练，10%验证）
4. 合并数据集并随机打乱

## 使用方法

### 环境要求

- Python 3.6+
- pandas
- numpy
- scikit-learn

### 运行数据处理

```bash
# 处理并合并数据集
python3 scripts/process_datasets.py
```

处理后的数据集将保存在 `data/processed/combined_sms_dataset.csv`，统计信息将保存在 `data/processed/dataset_stats.md`。

## 数据集统计

- 总样本数: 15,574
- OTP短信: 10,000 (64.2%)
- 非OTP短信: 5,574 (35.8%)

### 训练集/验证集划分

- 训练集: 14,016 (90.0%)
  - OTP短信: 9,000
  - 非OTP短信: 5,016

- 验证集: 1,558 (10.0%)
  - OTP短信: 1,000
  - 非OTP短信: 558 # otp_model_for_golang
