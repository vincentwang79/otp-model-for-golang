# Go OTP检测器使用指南

本文档描述了如何在Go应用程序中使用训练好的OTP检测模型。

## 1. 模型文件

训练好的模型参数存储在以下文件中：
- `models/go_params/otp_svm_params.json`: SVM模型参数，包括特征权重和截距
- `models/go_params/otp_tfidf_params.json`: 文本向量化参数，包括词汇表

## 2. Go检测器结构

OTP检测器的Go实现位于：`src/go/detector/detector.go`

主要组件：
- `ModelParams`: 存储从Python导出的模型参数的结构
- `OTPDetector`: 实现OTP检测功能的主要结构
- 文本预处理函数：清洗文本、提取特征等

## 3. 在Go应用程序中使用检测器

### 3.1 基本用法

```go
package main

import (
    "fmt"
    "log"
    
    otpdetector "github.com/yourusername/otp-detector/src/go/detector"
)

func main() {
    // 创建检测器实例
    detector, err := otpdetector.NewOTPDetector("path/to/otp_svm_params.json")
    if err != nil {
        log.Fatalf("创建检测器失败: %v", err)
    }
    
    // 检测消息
    message := "Your verification code is 123456. Valid for 5 minutes."
    isOTP, confidence, err := detector.IsOTP(message)
    if err != nil {
        log.Fatalf("检测失败: %v", err)
    }
    
    fmt.Printf("消息: %s\n", message)
    fmt.Printf("是否为OTP: %v, 置信度: %.4f\n", isOTP, confidence)
}
```

### 3.2 使用CLI工具

项目提供了一个CLI工具，用于测试模型：`test/cli/main.go`

运行CLI工具：

```bash
cd test/cli
go run main.go
```

CLI工具支持以下模式：
- 默认模式：使用预定义的示例测试模型
- 交互模式：`go run main.go -interactive`
- 文件处理模式：`go run main.go -file path/to/messages.txt`
- 基准测试模式：`go run main.go -benchmark`

### 3.3 调试模式

可以启用调试模式，查看更多信息：

```go
detector.EnableDebug(true)
```

## 4. 集成到现有应用程序

### 4.1 导入依赖

如果您的应用程序使用Go模块，请将detector包添加为依赖：

```bash
go get github.com/yourusername/otp-detector/src/go/detector
```

或者直接复制`detector.go`文件到您的项目中。

### 4.2 初始化检测器

在应用程序启动时初始化检测器：

```go
var detector *otpdetector.OTPDetector

func init() {
    var err error
    detector, err = otpdetector.NewOTPDetector("path/to/otp_svm_params.json")
    if err != nil {
        log.Fatalf("初始化OTP检测器失败: %v", err)
    }
}
```

### 4.3 在处理短信的代码中使用

```go
func processMessage(message string) {
    isOTP, confidence, err := detector.IsOTP(message)
    if err != nil {
        log.Printf("OTP检测失败: %v", err)
        return
    }
    
    if isOTP {
        // 处理OTP短信
        handleOTPMessage(message, confidence)
    } else {
        // 处理非OTP短信
        handleNonOTPMessage(message)
    }
}
```

## 5. 性能考虑

- 检测器在初始化时会加载整个模型到内存中，这可能会占用一些内存
- 每次调用`IsOTP`方法都会进行文本预处理和特征提取，这可能会消耗一些CPU资源
- 对于高并发应用，建议使用单例模式或对象池来管理检测器实例

## 6. 故障排除

如果遇到问题，请检查：

1. 模型参数文件路径是否正确
2. JSON文件格式是否有效
3. 启用调试模式，查看更多信息：`detector.EnableDebug(true)`

## 7. 示例代码

完整的示例代码可以在`test/cli/main.go`中找到，它展示了如何：

1. 初始化检测器
2. 处理单条消息
3. 批量处理消息
4. 处理不同语言的消息（英文、中文等） 