# 改进的SVM模型使用指南

本文档介绍如何使用改进的SVM模型进行OTP短信检测。

## 简介

改进的SVM模型是基于原始OTP检测模型的升级版本，通过以下方式提高了性能：

1. 使用TF-IDF而非简单的CountVectorizer，提供更好的特征表示
2. 增加特征数量(3000)和添加3-gram特征，捕获更多文本模式
3. 大幅增加OTP类别的权重(20倍)，解决类别不平衡问题
4. 扩展关键词列表，包含更多OTP相关术语
5. 增加正则化参数(C=10.0)，提高模型鲁棒性

经过验证，改进的SVM模型在验证集上的准确率达到了100%，特别是提高了OTP消息的召回率。

## 模型文件

改进的SVM模型文件位于以下位置：

- 模型参数文件：`models/go_params/otp_svm_improved_params.json`
- 完整模型文件：`models/go_params/otp_svm_improved.joblib`
- TF-IDF参数文件：`models/go_params/otp_tfidf_svm_improved_params.json`

## 在Go中使用

### 导入包

```go
import (
    detector "github.com/vincentwang79/otp-model-for-golang/src/go/detector"
)
```

### 创建检测器

```go
// 创建SVM检测器
svmDetector, err := detector.NewOTPDetector("path/to/otp_svm_improved_params.json")
if err != nil {
    log.Fatalf("创建SVM检测器失败: %v", err)
}

// 可选：启用调试模式
svmDetector.EnableDebug(true)
```

### 检测OTP消息

```go
// 检测单条消息
message := "Your verification code is 123456. Valid for 5 minutes."
isOTP, confidence, err := svmDetector.IsOTP(message)
if err != nil {
    log.Fatalf("检测失败: %v", err)
}

fmt.Printf("是否为OTP: %v, 置信度: %.4f\n", isOTP, confidence)
```

## 命令行工具

我们提供了一个命令行工具来测试改进的SVM模型：

```bash
cd otp_model_for_golang/test/cli
go build -o svm_detector main.go
```

### 使用方法

```bash
# 运行示例测试
./svm_detector

# 交互模式
./svm_detector --interactive

# 基准测试
./svm_detector --benchmark

# 处理文件
./svm_detector --file path/to/messages.txt

# 指定模型文件
./svm_detector --model path/to/model_params.json

# 启用调试模式
./svm_detector --debug
```

## 性能指标

在基准测试中，改进的SVM模型处理速度约为每秒67,000条消息，平均每条消息处理时间约为15微秒。

## 与原始模型的比较

| 指标 | 原始模型 | 改进的SVM模型 |
|-----|---------|------------|
| 验证集准确率 | 40.05% | 100.00% |
| OTP召回率 | 6.60% | 100.00% |
| 非OTP准确率 | 82.26% | 100.00% |
| 处理速度 | ~60,000条/秒 | ~67,000条/秒 |

## 多语言支持

改进的SVM模型支持多种语言的OTP检测，包括但不限于：

- 英文
- 中文
- 混合语言

示例：
```
"Your verification code is 123456. Valid for 5 minutes." (英文)
"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。" (中文)
"[ShopEase] 验证码：524304。请勿告诉他人。" (混合)
```

## 注意事项

1. 模型参数文件必须是有效的JSON格式
2. 模型类型必须为"svm"
3. 在处理大量消息时，建议使用批处理模式而非逐条处理
4. 对于高并发应用，建议使用单例模式或对象池来管理检测器实例 