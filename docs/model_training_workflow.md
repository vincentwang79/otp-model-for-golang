# OTP检测模型训练工作流

本文档描述了如何训练OTP（一次性密码）检测模型并将其导入到Go应用程序中使用的完整工作流。

## 1. 数据准备

训练数据存储在CSV文件中，格式如下：
- `message`: 短信文本内容
- `is_otp`: 标签，1表示OTP消息，0表示非OTP消息
- `split`: 数据集划分，'train'表示训练集，'val'表示验证集

数据集位置：`data/processed/combined_sms_dataset.csv`

数据集统计信息：
- 总样本数: 15574
- OTP短信: 10000 (64.2%)
- 非OTP短信: 5574 (35.8%)
- 训练集: 14016 (90.0%)
- 验证集: 1558 (10.0%)

## 2. 环境设置

```bash
# 进入项目目录
cd otp_model_for_golang

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 3. 模型训练

训练脚本位置：`src/python/train_export_model.py`

该脚本执行以下操作：
1. 加载训练数据（仅使用标记为'train'的数据）
2. 预处理文本，提取特征
3. 训练线性SVM分类器
4. 导出模型参数为JSON格式，便于Go程序使用
5. 保存完整的Python模型（用于后续比较或改进）

```bash
# 运行训练脚本
python src/python/train_export_model.py
```

## 4. 模型输出

训练完成后，以下文件将被保存到`models/go_params/`目录：
- `otp_svm_params.json`: SVM模型参数，包括特征权重和截距
- `otp_tfidf_params.json`: 文本向量化参数，包括词汇表和配置
- `otp_svm.joblib`: 完整的Python模型（用于Python环境中的使用）
- `processed_examples.txt`: 一些处理后的文本示例（用于调试）

## 5. 模型测试

可以使用CLI工具测试训练好的模型：

```bash
cd test/cli
go run main.go
```

CLI工具支持以下模式：
- 默认模式：使用预定义的示例测试模型
- 交互模式：`go run main.go -interactive`
- 文件处理模式：`go run main.go -file path/to/messages.txt`
- 基准测试模式：`go run main.go -benchmark`

## 6. 模型改进

如需改进模型，可以：
1. 调整`train_export_model.py`中的模型参数
2. 增加训练数据或改进数据质量
3. 尝试不同的特征提取方法
4. 尝试不同的分类算法

每次改进后，重新运行训练脚本并测试模型性能。 