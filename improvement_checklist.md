# OTP检测模型改进清单

## 数据平衡问题
- [x] 使用SMOTE技术对OTP类别进行过采样
- [x] 对非OTP类别进行欠采样，平衡类别比例
- [x] 调整训练数据中OTP与非OTP的比例至接近1:1
- [x] 结合增强特征工程和数据平衡策略训练模型

## 特征工程优化
- [x] 增强数字模式识别特征
- [x] 调整OTP关键词列表及其权重
- [x] 为中文短信添加专门的特征提取
- [x] 考虑使用词嵌入(Word Embeddings)替代TF-IDF
- [x] 优化n-gram特征的选择

## 模型参数调整
- [x] 重新调整SVM的类别权重参数
- [x] 优化朴素贝叶斯模型参数(当前表现最佳)
- [x] 尝试调整随机森林的树数量和深度
- [ ] 进行交叉验证以找到最佳参数组合

## 决策阈值优化
- [x] 基于验证集上的性能调整最佳决策阈值
- [x] 为不同语言(英文/中文)设置不同的决策阈值
- [x] 实现基于ROC曲线的最佳阈值选择
- [x] 基于F1分数选择最佳决策阈值

## 多语言支持增强
- [x] 添加中文分词支持
- [x] 为不同语言创建专门的特征提取器
- [x] 考虑使用语言检测来应用不同的模型策略
- [x] 支持俄语和爱沙尼亚语

## 评估与监控
- [x] 创建更全面的测试集，包含各种语言和格式的短信
- [ ] 实现模型性能监控机制
- [ ] 设计A/B测试框架评估改进效果

## 代码优化
- [x] 重构特征提取代码，提高可扩展性
- [x] 优化Go实现的性能
- [x] 添加更详细的调试输出选项 
- [x] 实现多种平衡方法和模型类型的组合 