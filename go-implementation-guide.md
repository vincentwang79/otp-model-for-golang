# Go Implementation Guide for OTP Detector

This guide explains how to use the OTP (One-Time Password) detector in your Go applications by importing this repository as a package.

## Overview

The OTP detector is a machine learning-based tool that can identify whether a text message contains a one-time password (OTP). The detector is implemented in Go and uses model parameters exported from a Python-trained model.

The OTP detector has several implementations:
- `BalancedEnhancedOTPDetector` - The recommended detector with balanced data and enhanced features (recommended)
- `ImprovedOTPDetector` - An improved version of the basic detector
- `OTPDetector` - The basic implementation

## Installation

Add this repository to your Go project:

```bash
go get github.com/vincentwang79/otp-model-for-golang
```

## Basic Usage

Here's a simple example of how to use the recommended OTP detector in your Go code:

```go
package main

import (
	"fmt"
	"log"
	
	detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
	// Create a new balanced enhanced OTP detector (recommended)
	balancedEnhancedDetector, err := detector.NewBalancedEnhancedOTPDetector("models/go_params/otp_svm_balanced_enhanced_params.json")
	if err != nil {
		log.Fatalf("Failed to create OTP detector: %v", err)
	}
	
	// Enable debug mode (optional)
	balancedEnhancedDetector.EnableDebug(true)
	
	// Example message
	message := "Your verification code is 123456. Please enter it to complete the login."
	
	// Check if the message is an OTP
	isOTP := balancedEnhancedDetector.IsOTP(message)
	
	fmt.Printf("Message: %s\n", message)
	fmt.Printf("Is OTP: %v\n", isOTP)
	
	// If you need more detailed information, you can use the GetOTPScore method
	score, debugInfo := balancedEnhancedDetector.GetOTPScore(message)
	fmt.Printf("OTP Score: %.4f (Threshold: %.4f)\n", score, balancedEnhancedDetector.GetDecisionThreshold())
}
```

## Model Parameter Files

The repository includes pre-trained model parameter files in the `models/go_params/` directory:

- `otp_svm_balanced_enhanced_params.json` - Parameters for the balanced enhanced detector (recommended)
- `otp_nb_balanced_enhanced_params.json` - Parameters for the balanced enhanced detector with Naive Bayes
- `otp_svm_improved_params.json` - Parameters for the improved detector
- `otp_tfidf_svm_improved_params.json` - Parameters for the improved detector with TF-IDF features

默认情况下，模型参数文件位于项目根目录的 `models/go_params/` 目录中。如果您使用 `go get` 安装了这个包，模型参数文件会位于 Go 模块缓存中，您需要指定完整路径或将模型参数文件复制到您的项目中。

## API Reference

### BalancedEnhancedOTPDetector (Recommended)

```go
// Create a new balanced enhanced OTP detector
func NewBalancedEnhancedOTPDetector(paramsPath string) (*BalancedEnhancedOTPDetector, error)

// Enable or disable debug mode
func (d *BalancedEnhancedOTPDetector) EnableDebug(enable bool)

// Check if a message is an OTP
func (d *BalancedEnhancedOTPDetector) IsOTP(message string) bool

// Get the OTP score and debug information
func (d *BalancedEnhancedOTPDetector) GetOTPScore(message string) (float64, map[string]interface{})

// Get the decision threshold
func (d *BalancedEnhancedOTPDetector) GetDecisionThreshold() float64
```

### ImprovedOTPDetector

```go
// Create a new improved OTP detector
func NewImprovedOTPDetector(paramsPath string) (*ImprovedOTPDetector, error)

// Enable or disable debug mode
func (d *ImprovedOTPDetector) EnableDebug(enable bool)

// Check if a message is an OTP
func (d *ImprovedOTPDetector) IsOTP(message string) bool

// Get the OTP score and debug information
func (d *ImprovedOTPDetector) GetOTPScore(message string) (float64, map[string]interface{})
```

### OTPDetector (Basic)

```go
// Create a new OTP detector
func NewOTPDetector(paramsPath string) (*OTPDetector, error)

// Enable or disable debug mode
func (d *OTPDetector) EnableDebug(enable bool)

// Check if a message is an OTP
func (d *OTPDetector) IsOTP(message string) bool

// Get the OTP score and debug information
func (d *OTPDetector) GetOTPScore(message string) (float64, map[string]interface{})
```

## Processing Multiple Messages

Here's an example of how to process multiple messages:

```go
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	
	detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
	// Create a new detector
	otpDetector, err := detector.NewBalancedEnhancedOTPDetector("models/go_params/otp_svm_balanced_enhanced_params.json")
	if err != nil {
		log.Fatalf("Failed to create detector: %v", err)
	}
	
	// Open a file with messages
	file, err := os.Open("messages.txt")
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()
	
	scanner := bufio.NewScanner(file)
	
	// Process each message
	for scanner.Scan() {
		message := scanner.Text()
		if message == "" {
			continue
		}
		
		isOTP := otpDetector.IsOTP(message)
		score, _ := otpDetector.GetOTPScore(message)
		
		fmt.Printf("Message: %s\n", message)
		fmt.Printf("Is OTP: %v, Score: %.4f\n\n", isOTP, score)
	}
	
	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading file: %v", err)
	}
}
```

## Performance Considerations

The OTP detector is designed to be very efficient:
- Processing speed: ~100,000 messages per second
- Average processing time: ~10 microseconds per message

For high-throughput applications, consider:
1. Using goroutines to process messages in parallel
2. Pre-loading the detector in memory
3. Using the balanced enhanced detector for better accuracy

### 高并发处理示例

以下是一个高并发处理示例，使用工作池处理大量消息：

```go
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	
	detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

// 消息处理结果
type Result struct {
	Message string
	IsOTP   bool
	Score   float64
}

func main() {
	// 获取当前目录
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("无法获取当前目录: %v", err)
	}
	
	// 模型参数文件路径
	modelPath := filepath.Join(currentDir, "models", "go_params", "otp_svm_balanced_enhanced_params.json")
	
	// 创建OTP检测器
	log.Println("正在加载OTP检测模型...")
	otpDetector, err := detector.NewBalancedEnhancedOTPDetector(modelPath)
	if err != nil {
		log.Fatalf("创建OTP检测器失败: %v", err)
	}
	
	// 示例消息
	messages := []string{
		"Your verification code is 123456. Please enter it to complete the login.",
		"Your OTP for bank transaction is 987654. Valid for 5 minutes.",
		"Hi, how are you doing today? Let's meet for coffee tomorrow.",
		// ... 更多消息
	}
	
	// 并发处理消息
	results := processMessagesInParallel(otpDetector, messages, 10)
	
	// 处理结果
	for _, result := range results {
		fmt.Printf("消息: %s\n", result.Message)
		fmt.Printf("是否为OTP: %v, 分数: %.4f\n\n", result.IsOTP, result.Score)
	}
}

// 并发处理消息
func processMessagesInParallel(detector *detector.BalancedEnhancedOTPDetector, messages []string, workerCount int) []Result {
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

// 工作协程
func worker(detector *detector.BalancedEnhancedOTPDetector, jobs <-chan string, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()
	
	for message := range jobs {
		isOTP := detector.IsOTP(message)
		score, _ := detector.GetOTPScore(message)
		results <- Result{
			Message: message,
			IsOTP:   isOTP,
			Score:   score,
		}
	}
}
```

## Troubleshooting

Common issues:

1. **Model file not found**: Ensure the path to the model parameter file is correct
2. **Unsupported model type**: The detector only supports SVM and NB models as specified in the model parameters
3. **Low confidence scores**: Make sure you're using the latest model parameters

## 完整Web服务示例

以下是一个完整的Web服务示例，展示如何在HTTP API中使用OTP检测器：

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	
	detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

// 请求结构体
type DetectionRequest struct {
	Message string `json:"message"`
}

// 响应结构体
type DetectionResponse struct {
	IsOTP  bool    `json:"is_otp"`
	Score  float64 `json:"score"`
	Lang   string  `json:"lang,omitempty"`
}

var otpDetector *detector.BalancedEnhancedOTPDetector

func main() {
	// 获取当前目录
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("无法获取当前目录: %v", err)
	}
	
	// 模型参数文件路径
	modelPath := filepath.Join(currentDir, "models", "go_params", "otp_svm_balanced_enhanced_params.json")
	
	// 检查模型文件是否存在
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("模型文件不存在: %s", modelPath)
	}
	
	// 创建OTP检测器
	log.Println("正在加载OTP检测模型...")
	otpDetector, err = detector.NewBalancedEnhancedOTPDetector(modelPath)
	if err != nil {
		log.Fatalf("创建OTP检测器失败: %v", err)
	}
	
	// 设置HTTP路由
	http.HandleFunc("/api/detect", handleDetection)
	http.HandleFunc("/health", handleHealth)
	
	// 启动HTTP服务器
	port := "8080"
	log.Printf("服务器启动在端口 %s...", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// 处理OTP检测请求
func handleDetection(w http.ResponseWriter, r *http.Request) {
	// 只允许POST方法
	if r.Method != http.MethodPost {
		http.Error(w, "只支持POST方法", http.StatusMethodNotAllowed)
		return
	}
	
	// 解析请求
	var req DetectionRequest
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, "无效的请求格式", http.StatusBadRequest)
		return
	}
	
	// 检查消息是否为空
	if req.Message == "" {
		http.Error(w, "消息不能为空", http.StatusBadRequest)
		return
	}
	
	// 检测OTP
	isOTP := otpDetector.IsOTP(req.Message)
	score, debugInfo := otpDetector.GetOTPScore(req.Message)
	
	// 构建响应
	resp := DetectionResponse{
		IsOTP: isOTP,
		Score: score,
	}
	
	// 从处理后的文本中提取语言标记
	if processedText, ok := debugInfo["processed_text"].(string); ok {
		if lang, found := extractLang(processedText); found {
			resp.Lang = lang
		}
	}
	
	// 返回JSON响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// extractLang 从处理后的文本中提取语言标记
func extractLang(text string) (string, bool) {
	// 在处理后的文本中查找LANG_标记
	if strings.Contains(text, "LANG_zh") {
		return "zh", true
	} else if strings.Contains(text, "LANG_en") {
		return "en", true
	} else if strings.Contains(text, "LANG_ru") {
		return "ru", true
	} else if strings.Contains(text, "LANG_et") {
		return "et", true
	}
	return "unknown", false
}

// 健康检查端点
func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}
```

使用curl测试此API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"message":"Your verification code is 123456"}' http://localhost:8080/api/detect
```

## 在现有项目中集成

要在现有项目中集成OTP检测器，可以采用以下步骤：

1. **添加依赖**

   在您的`go.mod`文件中添加依赖：

   ```go
   require github.com/vincentwang79/otp-model-for-golang v0.0.0
   
   // 如果您是从本地路径引用
   replace github.com/vincentwang79/otp-model-for-golang => /path/to/otp_model_for_golang
   ```

2. **复制模型参数文件**

   将`models/go_params/otp_svm_balanced_enhanced_params.json`文件复制到您项目的适当位置。

3. **创建检测器单例**

   在应用程序启动时创建检测器实例，并在需要时重复使用：

   ```go
   package detector

   import (
       "sync"
       
       otpdetector "github.com/vincentwang79/otp-model-for-golang/detector"
   )

   var (
       instance *otpdetector.BalancedEnhancedOTPDetector
       once     sync.Once
       initErr  error
   )

   // GetOTPDetector 返回OTP检测器的单例实例
   func GetOTPDetector(modelPath string) (*otpdetector.BalancedEnhancedOTPDetector, error) {
       once.Do(func() {
           instance, initErr = otpdetector.NewBalancedEnhancedOTPDetector(modelPath)
       })
       return instance, initErr
   }
   ```

4. **在业务逻辑中使用**

   ```go
   // 获取检测器实例
   detector, err := detector.GetOTPDetector("path/to/model_params.json")
   if err != nil {
       // 处理错误
   }
   
   // 检测消息
   isOTP := detector.IsOTP(message)
   score, _ := detector.GetOTPScore(message)
   ```

## 总结

OTP检测器是一个高性能、易于集成的Go库，可用于检测文本消息中是否包含一次性密码(OTP)。主要特点包括：

- 高准确率：基于机器学习模型，准确率接近90%
- 高性能：每秒可处理约10万条消息
- 易于集成：简单的API，只需几行代码即可使用
- 支持多语言：可以检测英文、中文、俄语和爱沙尼亚语的OTP消息

通过本指南，您应该能够轻松地在您的Go应用程序中集成OTP检测功能。

## For LLM Developers

If you're a language model like Claude-3.7 working with this code:

1. The detector is implemented in the `detector` package
2. The recommended type is `BalancedEnhancedOTPDetector`
3. The detector uses the `IsOTP` method to check if a message contains an OTP
4. The model parameters are stored in JSON files
5. The model works by extracting text features, including n-grams, digit patterns, and OTP keywords
6. The detector supports SVM and Naive Bayes models with parameters exported from Python
7. The detector is thread-safe and can be used in concurrent environments
