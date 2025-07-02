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

### 3.1 原始模型训练

训练脚本位置：`src/python/train_export_model.py`

该脚本执行以下操作：
1. 加载训练数据（仅使用标记为'train'的数据）
2. 预处理文本，提取特征
3. 训练线性SVM分类器
4. 导出模型参数为JSON格式，便于Go程序使用
5. 保存完整的Python模型（用于后续比较或改进）

```bash
# 运行原始训练脚本
python src/python/train_export_model.py
```

### 3.2 改进模型训练

改进的训练脚本位置：`src/python/train_improved_model.py`

该脚本实现了以下改进：
1. 使用TF-IDF而非简单的CountVectorizer，提供更好的特征表示
2. 增加特征数量(3000)和添加3-gram特征，捕获更多文本模式
3. 大幅增加OTP类别的权重(20倍)，解决类别不平衡问题
4. 扩展关键词列表，包含更多OTP相关术语
5. 增加正则化参数(C=10.0)，提高模型鲁棒性

```bash
# 运行改进的训练脚本（默认使用SVM模型）
python src/python/train_improved_model.py

# 可以指定不同的模型类型（svm、nb、rf）
python src/python/train_improved_model.py --model svm
python src/python/train_improved_model.py --model nb  # 朴素贝叶斯
python src/python/train_improved_model.py --model rf  # 随机森林

# 指定自定义数据路径和输出目录
python src/python/train_improved_model.py --data path/to/data.csv --output path/to/output
```

## 4. 模型输出

### 4.1 原始模型输出

训练完成后，以下文件将被保存到`models/go_params/`目录：
- `otp_svm_params.json`: SVM模型参数，包括特征权重和截距
- `otp_tfidf_params.json`: 文本向量化参数，包括词汇表和配置
- `otp_svm.joblib`: 完整的Python模型（用于Python环境中的使用）
- `processed_examples.txt`: 一些处理后的文本示例（用于调试）

### 4.2 改进模型输出

改进的训练脚本会输出以下文件：
- `otp_svm_improved_params.json`: 改进的SVM模型参数
- `otp_svm_improved.joblib`: 完整的改进SVM模型
- `otp_tfidf_svm_improved_params.json`: 改进的TF-IDF参数

如果使用其他模型类型，将生成相应的文件：
- 朴素贝叶斯: `otp_nb_improved_params.json`, `otp_nb_improved.joblib`
- 随机森林: `otp_rf_improved_params.json`, `otp_rf_improved.joblib`

## 5. 模型验证

可以使用验证脚本评估模型性能：

```bash
# 验证原始模型
python src/python/validate_model.py

# 验证改进的模型
python src/python/validate_improved_model.py

# 验证特定模型类型
python src/python/validate_improved_model.py --model svm
```

验证脚本会输出以下指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 混淆矩阵

## 6. 模型测试

可以使用CLI工具测试训练好的模型：

```bash
cd test/cli
go run main.go
```

CLI工具支持以下模式：
- 默认模式：使用预定义的示例测试模型
- 交互模式：`go run main.go --interactive`
- 文件处理模式：`go run main.go --file path/to/messages.txt`
- 基准测试模式：`go run main.go --benchmark`
- 调试模式：`go run main.go --debug`
- 自定义模型：`go run main.go --model path/to/model_params.json`

## 7. 模型改进

如需进一步改进模型，可以：

1. 调整特征提取方法：
   - 修改n-gram范围
   - 调整TF-IDF参数
   - 添加新的特征类型

2. 调整模型参数：
   - 修改正则化参数C
   - 调整类别权重
   - 尝试不同的核函数（目前仅支持线性核）

3. 尝试不同的分类算法：
   - 朴素贝叶斯（适合文本分类）
   - 随机森林（更复杂但可能更准确）
   - 深度学习模型（需要更多数据）

4. 增强数据集：
   - 添加更多OTP和非OTP样本
   - 平衡类别分布
   - 增加多语言样本

每次改进后，重新运行训练脚本并使用验证脚本评估模型性能。

## 8. 模型部署

将训练好的模型部署到Go应用程序：

1. 复制模型参数文件到应用程序可访问的位置
2. 在Go应用程序中加载模型参数
3. 创建OTP检测器实例
4. 使用检测器处理消息

详细的部署指南请参考 `go_model_usage_guide.md`。 