# Go OTP检测器使用指南

本文档描述了如何在Go应用程序中使用训练好的OTP检测模型。

## 1. 模型文件

训练好的模型参数存储在以下文件中：
- `models/go_params/otp_svm_params.json`: 原始SVM模型参数
- `models/go_params/otp_svm_improved_params.json`: 改进的SVM模型参数
- `models/go_params/otp_tfidf_svm_improved_params.json`: 改进的TF-IDF参数
- `models/go_params/otp_svm_smote_params.json`: 使用SMOTE平衡的SVM模型参数
- `models/go_params/otp_nb_smote_params.json`: 使用SMOTE平衡的朴素贝叶斯模型参数
- `models/go_params/otp_rf_smote_params.json`: 使用SMOTE平衡的随机森林模型参数
- `models/go_params/otp_svm_balanced_enhanced_params.json`: 平衡增强SVM模型参数（推荐）
- `models/go_params/otp_nb_balanced_enhanced_params.json`: 平衡增强朴素贝叶斯模型参数
- `models/go_params/otp_rf_balanced_enhanced_params.json`: 平衡增强随机森林模型参数

## 2. Go检测器结构

OTP检测器的Go实现位于以下文件：
- `src/go/detector/detector.go`: 原始检测器
- `src/go/detector/detector_improved.go`: 改进的检测器
- `src/go/detector/detector_balanced_enhanced.go`: 平衡增强检测器（推荐）
- `src/go/detector/language_detector.go`: 语言检测器

主要组件：
- `ModelParams`: 存储从Python导出的模型参数的结构
- `OTPDetector`: 实现OTP检测功能的主要结构
- `ImprovedOTPDetector`: 改进的OTP检测器
- `BalancedEnhancedOTPDetector`: 平衡增强OTP检测器
- `LanguageDetector`: 语言检测器
- 文本预处理函数：清洗文本、提取特征、识别数字模式和关键词等

## 3. 在Go应用程序中使用检测器

### 3.1 基本用法

```go
package main

import (
    "fmt"
    "log"
    
    detector "github.com/vincentwang79/otp-model-for-golang/src/go/detector"
)

func main() {
    // 创建检测器实例 - 选择一种检测器
    
    // 1. 原始检测器
    otpDetector, err := detector.NewOTPDetector("path/to/otp_svm_params.json")
    
    // 2. 改进的检测器
    improvedDetector, err := detector.NewImprovedOTPDetector("path/to/otp_svm_improved_params.json")
    
    // 3. 平衡增强检测器（推荐）
    balancedEnhancedDetector, err := detector.NewBalancedEnhancedOTPDetector("path/to/otp_svm_balanced_enhanced_params.json")
    
    if err != nil {
        log.Fatalf("创建检测器失败: %v", err)
    }
    
    // 检测消息
    message := "Your verification code is 123456. Valid for 5 minutes."
    
    // 使用原始检测器
    isOTP, confidence, err := otpDetector.IsOTP(message)
    
    // 使用改进的检测器
    isOTP, confidence, err := improvedDetector.IsOTP(message)
    
    // 使用平衡增强检测器（推荐）
    isOTP := balancedEnhancedDetector.IsOTP(message)
    
    if err != nil {
        log.Fatalf("检测失败: %v", err)
    }
    
    fmt.Printf("消息: %s\n", message)
    fmt.Printf("是否为OTP: %v, 置信度: %.4f\n", isOTP, confidence)
}
```

### 3.2 使用CLI工具

项目提供了多个CLI工具，用于测试不同的模型：

1. 原始模型CLI: `test/cli/main.go`
2. 改进模型CLI: `test/cli/main_improved.go`
3. 平衡增强模型CLI: `test/balanced_enhanced_cli/main.go`（推荐）

运行平衡增强模型CLI工具：

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

### 3.3 调试模式

可以启用调试模式，查看更多信息：

```go
detector.EnableDebug(true)
```

调试模式会输出以下信息：
- 输入文本
- 处理后的文本（包含提取的特征）
- 分类分数和置信度
- 最终判断结果
- 检测到的语言（仅限平衡增强检测器）

## 4. 集成到现有应用程序

### 4.1 导入依赖

如果您的应用程序使用Go模块，请将detector包添加为依赖：

```bash
go get github.com/vincentwang79/otp-model-for-golang/src/go/detector
```

或者直接复制相关文件到您的项目中。

### 4.2 初始化检测器

在应用程序启动时初始化检测器：

```go
var detector *detector.BalancedEnhancedOTPDetector

func init() {
    var err error
    detector, err = detector.NewBalancedEnhancedOTPDetector("path/to/otp_svm_balanced_enhanced_params.json")
    if err != nil {
        log.Fatalf("初始化OTP检测器失败: %v", err)
    }
}
```

### 4.3 在处理短信的代码中使用

```go
func processMessage(message string) {
    isOTP := detector.IsOTP(message)
    
    if isOTP {
        // 处理OTP短信
        handleOTPMessage(message)
    } else {
        // 处理非OTP短信
        handleNonOTPMessage(message)
    }
}
```

### 4.4 高并发处理

对于需要处理大量消息的应用，可以使用goroutines并行处理：

```go
func processMessagesInParallel(messages []string, workerCount int) []Result {
    var wg sync.WaitGroup
    
    // 创建工作通道
    jobs := make(chan string, len(messages))
    results := make(chan Result, len(messages))
    
    // 启动工作协程
    for i := 0; i < workerCount; i++ {
        wg.Add(1)
        go worker(detector, jobs, results, &wg)
    }
    
    // 发送消息到工作通道
    for _, message := range messages {
        jobs <- message
    }
    close(jobs)
    
    // 等待所有工作完成
    wg.Wait()
    close(results)
    
    // 收集结果
    var resultList []Result
    for result := range results {
        resultList = append(resultList, result)
    }
    
    return resultList
}

func worker(detector *detector.BalancedEnhancedOTPDetector, jobs <-chan string, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    
    for message := range jobs {
        isOTP := detector.IsOTP(message)
        results <- Result{
            Message: message,
            IsOTP:   isOTP,
        }
    }
}
```

## 5. 性能考虑

- 检测器在初始化时会加载整个模型到内存中，这可能会占用一些内存（约2-3MB）
- 每次调用`IsOTP`方法都会进行文本预处理和特征提取，这可能会消耗一些CPU资源
- 对于高并发应用，建议使用单例模式或对象池来管理检测器实例
- 处理速度：每秒约67,000条消息（在标准硬件上）
- 每条消息平均处理时间：约15微秒
- 平衡增强模型的性能与原始模型相当，但准确率和F1分数更高

## 6. 故障排除

如果遇到问题，请检查：

1. 模型参数文件路径是否正确
2. JSON文件格式是否有效
3. 启用调试模式，查看更多信息：`detector.EnableDebug(true)`
4. 确保模型类型与参数文件匹配（svm、nb或rf）
5. 检查是否有足够的内存加载模型
6. 对于中文文本，确保已正确处理字符编码

## 7. 示例代码

完整的示例代码可以在以下文件中找到：

1. 原始模型：`test/cli/main.go`
2. 改进模型：`test/cli/main_improved.go`
3. 平衡增强模型：`test/balanced_enhanced_cli/main.go`

这些示例展示了如何：

1. 初始化检测器
2. 处理单条消息
3. 批量处理消息
4. 处理不同语言的消息（英文、中文、俄语、爱沙尼亚语等）
5. 运行基准测试

## 8. Web服务示例

以下是一个简单的HTTP API示例，展示如何在Web服务中使用平衡增强OTP检测器：

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    
    detector "github.com/vincentwang79/otp-model-for-golang/src/go/detector"
)

var otpDetector *detector.BalancedEnhancedOTPDetector

func init() {
    var err error
    otpDetector, err = detector.NewBalancedEnhancedOTPDetector("path/to/otp_svm_balanced_enhanced_params.json")
    if err != nil {
        log.Fatalf("初始化OTP检测器失败: %v", err)
    }
}

type Request struct {
    Message string `json:"message"`
}

type Response struct {
    IsOTP bool   `json:"is_otp"`
    Lang  string `json:"lang,omitempty"` // 检测到的语言
}

func handleDetect(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "仅支持POST请求", http.StatusMethodNotAllowed)
        return
    }
    
    var req Request
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "无效的请求体", http.StatusBadRequest)
        return
    }
    
    // 启用调试以获取语言信息
    otpDetector.EnableDebug(true)
    
    // 检测是否为OTP
    isOTP := otpDetector.IsOTP(req.Message)
    
    // 获取检测到的语言（在实际应用中，您可能需要修改detector以返回语言信息）
    lang := "unknown" // 这里仅为示例，实际应用中需要从检测器获取
    
    resp := Response{
        IsOTP: isOTP,
        Lang:  lang,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

func main() {
    http.HandleFunc("/api/detect", handleDetect)
    
    fmt.Println("启动OTP检测API服务，监听端口8080...")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

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