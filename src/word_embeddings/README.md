# 词嵌入OTP检测器 - Python模块

本目录包含用于训练和评估基于词嵌入的OTP检测模型的Python模块。

## 模块说明

- `word_embeddings.py` - 词嵌入模型评估器
- `word_embedding_feature_extractor.py` - 词嵌入特征提取器
- `word_embedding_model_trainer.py` - 词嵌入模型训练器

## 使用方法

### 安装依赖

```bash
# 激活虚拟环境
source ../../venv_py3/bin/activate

# 安装依赖
pip install gensim tensorflow transformers fasttext
pip install jieba pymorphy2 estnltk
```

### 评估词嵌入模型

```python
from word_embeddings import WordEmbeddingEvaluator

# 初始化评估器
evaluator = WordEmbeddingEvaluator("../../data/processed/balanced_smote_dataset.csv")

# 评估模型
result = evaluator.evaluate_model()
print(result)
```

### 提取词嵌入特征

```python
from word_embedding_feature_extractor import WordEmbeddingFeatureExtractor

# 初始化特征提取器
extractor = WordEmbeddingFeatureExtractor()

# 提取特征
texts = ["Your verification code is 123456", "Hello, how are you?"]
features = extractor.transform(texts)
print(features)
```

### 训练词嵌入模型

```python
from word_embedding_model_trainer import WordEmbeddingModelTrainer

# 初始化模型训练器
trainer = WordEmbeddingModelTrainer("../../data/processed/balanced_smote_dataset.csv", "svm")

# 训练模型
result = trainer.run()
print(result)
```

## 输出文件

模型训练后，将在`models`目录下生成以下文件：

- `word_embedding_vectors.json` - 词嵌入向量
- `word_embedding_svm_model.joblib` - 训练好的SVM模型
- `word_embedding_svm_threshold.txt` - 最佳决策阈值

这些文件将被Go实现的词嵌入检测器使用。 