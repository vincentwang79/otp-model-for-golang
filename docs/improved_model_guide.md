# 改进的OTP检测模型使用指南

本文档介绍如何使用改进的OTP短信检测模型。

## 简介

我们开发了多种改进的OTP检测模型，按照开发顺序和性能提升可分为：

1. **改进的SVM模型**：基于原始OTP检测模型的升级版本
2. **平衡数据模型**：解决数据集不平衡问题的模型
3. **平衡增强模型**：结合数据平衡和增强特征工程的最新模型（推荐）

### 改进的SVM模型

改进的SVM模型通过以下方式提高了性能：
- 使用TF-IDF而非简单的CountVectorizer，提供更好的特征表示
- 增加特征数量(3000)和添加3-gram特征，捕获更多文本模式
- 大幅增加OTP类别的权重(20倍)，解决类别不平衡问题
- 扩展关键词列表，包含更多OTP相关术语
- 增加正则化参数(C=10.0)，提高模型鲁棒性

### 平衡增强模型（推荐）

平衡增强模型在改进的SVM模型基础上进一步提升：
- 实现了SMOTE过采样和随机欠采样两种数据平衡策略
- 增强了多语言支持（英文、中文、俄语、爱沙尼亚语）
- 优化了数字模式识别和关键词权重
- 支持多种分类器：SVM、朴素贝叶斯(NB)和随机森林(RF)

## 模型文件

### 改进的SVM模型文件

- 模型参数文件：`models/go_params/otp_svm_improved_params.json`
- 完整模型文件：`models/go_params/otp_svm_improved.joblib`
- TF-IDF参数文件：`models/go_params/otp_tfidf_svm_improved_params.json`

### 平衡增强模型文件（推荐）

- SVM模型参数：`models/go_params/otp_svm_balanced_enhanced_params.json`
- 朴素贝叶斯模型参数：`models/go_params/otp_nb_balanced_enhanced_params.json`
- 随机森林模型参数：`models/go_params/otp_rf_balanced_enhanced_params.json`

## 在Go中使用

### 导入包

```go
import (
    detector "github.com/vincentwang79/otp-model-for-golang/src/go/detector"
)
```

### 创建检测器

#### 改进的SVM检测器

```go
// 创建改进的SVM检测器
improvedDetector, err := detector.NewImprovedOTPDetector("path/to/otp_svm_improved_params.json")
if err != nil {
    log.Fatalf("创建改进的SVM检测器失败: %v", err)
}

// 可选：启用调试模式
improvedDetector.EnableDebug(true)
```

#### 平衡增强检测器（推荐）

```go
// 创建平衡增强检测器（支持SVM、NB、RF三种模型类型）
balancedEnhancedDetector, err := detector.NewBalancedEnhancedOTPDetector("path/to/otp_svm_balanced_enhanced_params.json")
if err != nil {
    log.Fatalf("创建平衡增强检测器失败: %v", err)
}

// 可选：启用调试模式
balancedEnhancedDetector.EnableDebug(true)
```

### 检测OTP消息

#### 使用改进的SVM检测器

```go
// 检测单条消息
message := "Your verification code is 123456. Valid for 5 minutes."
isOTP, confidence, err := improvedDetector.IsOTP(message)
if err != nil {
    log.Fatalf("检测失败: %v", err)
}

fmt.Printf("是否为OTP: %v, 置信度: %.4f\n", isOTP, confidence)
```

#### 使用平衡增强检测器（推荐）

```go
// 检测单条消息
message := "Your verification code is 123456. Valid for 5 minutes."
isOTP := balancedEnhancedDetector.IsOTP(message)

fmt.Printf("是否为OTP: %v\n", isOTP)
```

## 命令行工具

我们提供了多个命令行工具来测试不同的模型：

### 改进的SVM模型CLI

```bash
cd otp_model_for_golang/test/cli
go build -o improved_detector main_improved.go
```

#### 使用方法

```bash
# 运行示例测试
./improved_detector

# 交互模式
./improved_detector --interactive

# 基准测试
./improved_detector --benchmark

# 处理文件
./improved_detector --file path/to/messages.txt

# 指定模型文件
./improved_detector --model path/to/model_params.json

# 启用调试模式
./improved_detector --debug
```

### 平衡增强模型CLI（推荐）

```bash
cd otp_model_for_golang/test/balanced_enhanced_cli
go build -o balanced_enhanced_detector main.go
```

#### 使用方法

```bash
# 运行示例测试
./balanced_enhanced_detector

# 交互模式
./balanced_enhanced_detector --interactive

# 基准测试
./balanced_enhanced_detector --benchmark

# 处理文件
./balanced_enhanced_detector --file path/to/messages.txt

# 指定模型文件
./balanced_enhanced_detector --model path/to/model_params.json

# 启用调试模式
./balanced_enhanced_detector --debug

# 选择模型类型
./balanced_enhanced_detector --type nb  # 支持svm、nb、rf

# 选择平衡方法
./balanced_enhanced_detector --balance smote  # 支持smote、undersample
```

## 性能指标

在基准测试中，各模型处理速度约为每秒67,000条消息，平均每条消息处理时间约为15微秒。

## 模型性能比较

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

## 多语言支持

所有改进的模型都支持多种语言的OTP检测，但平衡增强模型的多语言支持最为完善，包括：

- 英文
- 中文
- 俄语
- 爱沙尼亚语

示例：
```
"Your verification code is 123456. Valid for 5 minutes." (英文)
"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。" (中文)
"Ваш проверочный код: 123456. Действителен в течение 5 минут." (俄语)
"Teie kinnituskood on 123456. Kehtib 5 minutit." (爱沙尼亚语)
"[ShopEase] 验证码：524304。请勿告诉他人。" (混合)
```

## 注意事项

1. 模型参数文件必须是有效的JSON格式
2. 模型类型必须与参数文件匹配（svm、nb或rf）
3. 在处理大量消息时，建议使用批处理模式而非逐条处理
4. 对于高并发应用，建议使用单例模式或对象池来管理检测器实例
5. 平衡增强模型的`IsOTP`方法不返回错误，使用更简洁的接口
6. 对于中文文本，确保已正确处理字符编码 