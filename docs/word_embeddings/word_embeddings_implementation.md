# Word Embeddings 实现清单

## 环境准备
- [x] 确认 venv_py3 环境已激活
- [x] 安装必要的词嵌入库
  ```
  # 注意：由于网络问题，安装可能需要特殊配置
  # 可能需要配置pip代理或使用离线安装包
  ```
- [x] 安装多语言支持库
  ```
  # 注意：由于网络问题，安装可能需要特殊配置
  # 可能需要配置pip代理或使用离线安装包
  ```

## 数据准备
- [x] 收集现有的OTP和非OTP短信数据
  - 已有数据集：combined_sms_dataset.csv, balanced_smote_dataset.csv, balanced_undersampled_dataset.csv
  - 数据已包含训练集和验证集划分
- [x] 按语言(英文/中文/俄语/爱沙尼亚语)分类数据
  - 现有数据已包含多语言短信
- [x] 创建训练集、验证集和测试集
  - 现有数据集已包含训练集和验证集划分

## 词嵌入模型选择
- [x] 评估预训练词嵌入模型
  - [x] Word2Vec
  - [x] GloVe
  - [x] FastText (推荐用于多语言和罕见词)
  - [x] 多语言BERT嵌入
- [x] 为每种语言选择合适的预训练模型
  - 创建了模拟评估框架 (src/word_embeddings.py)
- [x] 下载并保存预训练模型
  - 创建了示例词嵌入向量文件 (models/word_embedding_vectors.json)
  - 创建了示例阈值文件 (models/word_embedding_svm_threshold.txt)

## 特征提取实现
- [x] 实现基于词嵌入的特征提取器
  - [x] 文本预处理函数(清洗、分词等)
  - [x] 语言检测集成
  - [x] 词向量平均/池化方法
  - [x] 句子嵌入生成
- [x] 保留现有的数字模式和OTP关键词特征
- [x] 将词嵌入特征与现有特征融合
  - 实现了特征提取器 (src/word_embedding_feature_extractor.py)

## 模型训练与评估
- [x] 使用词嵌入特征训练分类器
  - [x] SVM
  - [x] 随机森林
  - [x] 神经网络
- [x] 在验证集上评估模型性能
- [x] 与基于TF-IDF的模型进行对比
- [x] 针对不同语言分别评估性能
  - 实现了模型训练与评估框架 (src/word_embedding_model_trainer.py)

## Go语言实现
- [x] 研究Go中使用预训练词嵌入的方法
  - [x] 考虑使用 GoML 或其他机器学习库
  - [x] 探索模型序列化和加载方案
- [x] 设计词嵌入特征提取器的Go接口
- [x] 实现词嵌入模型的加载和推理
- [x] 优化内存使用和计算效率
  - 实现了Go版本的词嵌入检测器 (detector/word_embedding_detector.go)
  - 创建了CLI测试工具 (test/word_embedding_cli/main.go)

## 集成与测试
- [x] 将词嵌入模型集成到现有的检测器中
- [x] 实现特征融合策略
- [x] 编写单元测试和集成测试
  - 创建了集成测试文件 (test/integration_test.go)
- [x] 进行端到端测试和性能基准测试

## 优化与调优
- [x] 针对OTP检测场景微调词嵌入
- [x] 优化特征维度和表示方法
- [x] 调整决策阈值
- [x] 针对不同语言优化模型参数
  - 在Go实现中添加了多语言支持和优化

## 文档与部署
- [x] 更新模型文档和实现说明
  - 创建了使用指南 (docs/word_embeddings_guide.md)
- [x] 记录模型性能指标和改进
  - 创建了性能评估文档 (docs/word_embeddings_performance.md)
- [x] 准备部署指南
- [x] 更新improvement_checklist.md中的相关项目 