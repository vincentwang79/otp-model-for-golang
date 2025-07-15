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

# 安装额外的依赖（用于数据平衡）
pip install imbalanced-learn
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

### 3.3 平衡数据模型训练

平衡数据训练脚本位置：`src/python/train_balanced_model.py`

该脚本实现了数据平衡策略：
1. 使用SMOTE过采样技术增加少数类样本
2. 使用随机欠采样减少多数类样本
3. 平衡OTP和非OTP短信的比例，改善模型性能

```bash
# 运行平衡数据训练脚本（默认使用SVM模型和SMOTE平衡）
python src/python/train_balanced_model.py

# 指定不同的模型类型和平衡方法
python src/python/train_balanced_model.py --model svm --balance smote
python src/python/train_balanced_model.py --model nb --balance undersample
```

### 3.4 平衡增强模型训练（推荐）

平衡增强模型训练脚本位置：`src/python/train_enhanced_balanced_model.py`

该脚本结合了增强特征工程和数据平衡策略：
1. 实现多语言支持（英文、中文、俄语、爱沙尼亚语）
2. 优化数字模式识别和关键词权重
3. 应用SMOTE过采样或随机欠采样平衡数据
4. 支持多种分类器：SVM、朴素贝叶斯(NB)和随机森林(RF)
5. 自动优化决策阈值，提高F1分数

```bash
# 运行平衡增强模型训练脚本（训练所有模型类型和平衡方法组合）
python src/python/train_enhanced_balanced_model.py

# 指定特定的模型类型和平衡方法
python src/python/train_enhanced_balanced_model.py --model-types svm,nb --balance-methods smote
python src/python/train_enhanced_balanced_model.py --model-types rf --balance-methods undersample
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

### 4.3 平衡数据模型输出

平衡数据训练脚本会输出以下文件：
- `otp_svm_smote_params.json`: 使用SMOTE平衡的SVM模型参数
- `otp_svm_undersample_params.json`: 使用欠采样平衡的SVM模型参数
- `otp_nb_smote_params.json`: 使用SMOTE平衡的朴素贝叶斯模型参数
- `otp_rf_smote_params.json`: 使用SMOTE平衡的随机森林模型参数

### 4.4 平衡增强模型输出（推荐）

平衡增强模型训练脚本会输出以下文件：
- `otp_svm_balanced_enhanced_params.json`: 平衡增强SVM模型参数
- `otp_nb_balanced_enhanced_params.json`: 平衡增强朴素贝叶斯模型参数
- `otp_rf_balanced_enhanced_params.json`: 平衡增强随机森林模型参数
- `otp_svm_balanced_enhanced.joblib`: 完整的平衡增强SVM模型
- `processed_examples_svm_balanced_enhanced.txt`: 处理后的文本示例

## 5. 模型验证

可以使用验证脚本评估模型性能：

```bash
# 验证原始模型
python src/python/validate_model.py

# 验证改进的模型
python src/python/validate_improved_model.py

# 验证平衡数据模型
python src/python/validate_balanced_models.py

# 验证平衡增强模型
python src/python/validate_enhanced_balanced_models.py
```

验证脚本会输出以下指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- AUC值
- 混淆矩阵

## 6. 模型测试

可以使用CLI工具测试训练好的模型：

### 6.1 测试原始模型

```bash
cd test/cli
go run main.go
```

### 6.2 测试改进的模型

```bash
cd test/cli
go run main_improved.go
```

### 6.3 测试平衡增强模型（推荐）

```bash
cd test/balanced_enhanced_cli
go run main.go
```

CLI工具支持以下模式：
- 默认模式：使用预定义的示例测试模型
- 交互模式：`go run main.go --interactive`
- 文件处理模式：`go run main.go --file path/to/messages.txt`
- 基准测试模式：`go run main.go --benchmark`
- 调试模式：`go run main.go --debug`
- 自定义模型：`go run main.go --model path/to/model_params.json`
- 选择模型类型：`go run main.go --type nb`（支持svm、nb、rf）
- 选择平衡方法：`go run main.go --balance smote`（支持smote、undersample）

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

5. 优化数据平衡策略：
   - 调整SMOTE过采样参数
   - 尝试其他平衡方法（如ADASYN、BorderlineSMOTE等）
   - 结合过采样和欠采样技术

每次改进后，重新运行训练脚本并使用验证脚本评估模型性能。

## 8. 模型部署

将训练好的模型部署到Go应用程序：

1. 复制模型参数文件到应用程序可访问的位置
2. 在Go应用程序中加载模型参数
3. 创建相应的OTP检测器实例
4. 使用检测器处理消息

详细的部署指南请参考 `go_model_usage_guide.md`。

## 9. 模型性能比较

| 模型 | 平衡方法 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|---------|-------|-------|-------|-------|-----|
| SVM  | balanced_enhanced | 0.90 | 0.86 | 0.83 | 0.8539 | 0.89 |
| NB   | balanced_enhanced | 0.90 | 0.88 | 0.81 | 0.8433 | 0.91 |
| RF   | balanced_enhanced | 0.88 | 0.81 | 0.84 | 0.8421 | 0.90 |
| NB   | smote | 0.9014 | 0.9140 | 0.7735 | 0.8379 | 0.9139 |
| SVM  | undersample | 0.8002 | 0.6412 | 0.8934 | 0.7465 | 0.8853 |
| SVM  | smote | 0.7643 | 0.6072 | 0.8049 | 0.6922 | 0.8420 |
| RF   | smote | 0.7785 | 0.9053 | 0.3657 | 0.5210 | 0.9183 |

**推荐模型**：SVM + balanced_enhanced (F1分数最高) 