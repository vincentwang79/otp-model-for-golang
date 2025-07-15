# Go OTP检测器使用指南

本文档描述了如何在Go应用程序中使用训练好的OTP检测模型。

## 1. 模型文件

训练好的模型参数存储在以下文件中：
- `models/go_params/otp_svm_balanced_enhanced_params.json`: 平衡增强SVM模型参数（**推荐使用**）
- `models/go_params/otp_nb_balanced_enhanced_params.json`: 平衡增强朴素贝叶斯模型参数
- `models/go_params/otp_rf_balanced_enhanced_params.json`: 平衡增强随机森林模型参数
- `models/go_params/otp_svm_params.json`: 原始SVM模型参数
- `models/go_params/otp_svm_improved_params.json`: 改进的SVM模型参数
- `models/go_params/otp_tfidf_svm_improved_params.json`: 改进的TF-IDF参数
- `models/go_params/otp_svm_smote_params.json`: 使用SMOTE平衡的SVM模型参数
- `models/go_params/otp_nb_smote_params.json`: 使用SMOTE平衡的朴素贝叶斯模型参数
- `models/go_params/otp_rf_smote_params.json`: 使用SMOTE平衡的随机森林模型参数

## 2. Go检测器结构

OTP检测器的Go实现位于以下文件：
- `detector/detector_balanced_enhanced.go`: 平衡增强检测器（**推荐使用**）
- `detector/detector_improved.go`: 改进的检测器
- `detector/detector.go`: 原始检测器
- `detector/language_detector.go`: 语言检测器

主要组件：
- `BalancedEnhancedOTPDetector`: 平衡增强OTP检测器（**推荐使用**）
- `ImprovedOTPDetector`: 改进的OTP检测器
- `OTPDetector`: 实现OTP检测功能的主要结构
- `LanguageDetector`: 语言检测器
- 文本预处理函数：清洗文本、提取特征、识别数字模式和关键词等

## 3. 在Go应用程序中使用检测器

### 3.1 基本用法

```go
package main

import (
    "fmt"
    "log"
    
    detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
    // 创建平衡增强检测器（推荐）
    balancedEnhancedDetector, err := detector.NewBalancedEnhancedOTPDetector("models/go_params/otp_svm_balanced_enhanced_params.json")
    
    if err != nil {
        log.Fatalf("创建检测器失败: %v", err)
    }
    
    // 检测消息
    message := "Your verification code is 123456. Valid for 5 minutes."
    
    // 使用平衡增强检测器
    isOTP := balancedEnhancedDetector.IsOTP(message)
    
    fmt.Printf("消息: %s\n", message)
    fmt.Printf("是否为OTP: %v\n", isOTP)
    
    // 如果需要更详细的信息，可以使用GetOTPScore方法
    score, debugInfo := balancedEnhancedDetector.GetOTPScore(message)
    fmt.Printf("OTP分数: %.4f (阈值: %.4f)\n", score, balancedEnhancedDetector.GetDecisionThreshold())
}
```

### 3.2 使用CLI工具

项目提供了多个CLI工具，用于测试不同的模型：

1. 平衡增强模型CLI: `test/balanced_enhanced_cli/main.go`（**推荐**）
2. 改进模型CLI: `test/cli/main_improved.go`
3. 原始模型CLI: `test/cli/main.go`

运行平衡增强模型CLI工具：

```bash
cd test/balanced_enhanced_cli
go run main.go -model ../../models/go_params/otp_svm_balanced_enhanced_params.json -type svm
```

CLI工具支持以下模式：
- 默认模式：使用预定义的示例测试模型
- 交互模式：`go run main.go -interactive`
- 文件处理模式：`go run main.go -file path/to/messages.txt`
- 基准测试模式：`go run main.go -benchmark`
- 调试模式：`go run main.go -debug`
- 自定义模型：`go run main.go -model path/to/model_params.json`
- 选择模型类型：`go run main.go -type svm`（支持svm、nb、rf）
- 选择平衡方法：`go run main.go -balance balanced_enhanced`（支持balanced_enhanced、smote、undersample）

### 3.3 调试模式

可以启用调试模式，查看更多信息：

```go
detector.EnableDebug(true)
```

调试模式会输出以下信息：
- 输入文本
- 处理后的文本（包含提取的特征）
- 分类分数和决策阈值
- 最终判断结果
- 检测到的语言
- 模型类型和平衡方法

## 4. 集成到实际产品中

### 4.1 导入依赖

如果您的应用程序使用Go模块，请将detector包添加为依赖：

```bash
go get github.com/vincentwang79/otp-model-for-golang/detector
```

或者直接复制相关文件到您的项目中：
- `src/go/detector/detector_balanced_enhanced.go`
- `src/go/detector/language_detector.go`

### 4.2 单例模式初始化检测器

在实际产品中，建议使用单例模式初始化检测器，避免重复加载模型：

```go
package otpdetection

import (
    "sync"
    
    detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

var (
    instance *detector.BalancedEnhancedOTPDetector
    once     sync.Once
    initErr  error
)

// GetDetector 返回OTP检测器的单例实例
func GetDetector() (*detector.BalancedEnhancedOTPDetector, error) {
    once.Do(func() {
        instance, initErr = detector.NewBalancedEnhancedOTPDetector("models/go_params/otp_svm_balanced_enhanced_params.json")
    })
    return instance, initErr
}
```

### 4.3 在微服务架构中使用

对于微服务架构，可以创建专门的OTP检测服务：

```go
package main

import (
    "encoding/json"
    "log"
    "net/http"
    
    detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

var otpDetector *detector.BalancedEnhancedOTPDetector

func init() {
    var err error
    otpDetector, err = detector.NewBalancedEnhancedOTPDetector("models/go_params/otp_svm_balanced_enhanced_params.json")
    if err != nil {
        log.Fatalf("初始化OTP检测器失败: %v", err)
    }
}

type Request struct {
    Message string `json:"message"`
}

type Response struct {
    IsOTP   bool    `json:"is_otp"`
    Score   float64 `json:"score,omitempty"`
    Lang    string  `json:"lang,omitempty"`
    Message string  `json:"message,omitempty"`
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
    
    // 获取OTP分数和调试信息
    score, debugInfo := otpDetector.GetOTPScore(req.Message)
    isOTP := score >= otpDetector.GetDecisionThreshold()
    
    // 构建响应
    resp := Response{
        IsOTP: isOTP,
        Score: score,
    }
    
    // 如果有调试信息，添加到响应中
    if processedText, ok := debugInfo["processed_text"].(string); ok {
        resp.Message = processedText
    }
    
    // 从处理后的文本中提取语言标记
    if processedText, ok := debugInfo["processed_text"].(string); ok {
        if lang, found := extractLang(processedText); found {
            resp.Lang = lang
        }
    }
    
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

func main() {
    http.HandleFunc("/api/detect", handleDetect)
    
    log.Println("启动OTP检测API服务，监听端口8080...")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### 4.4 高并发处理

对于需要处理大量消息的产品，可以使用工作池模式：

```go
package otpprocessor

import (
    "sync"
    
    detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

// Result 表示OTP检测结果
type Result struct {
    Message string
    IsOTP   bool
    Score   float64
}

// Processor OTP消息处理器
type Processor struct {
    detector    *detector.BalancedEnhancedOTPDetector
    workerCount int
    jobChan     chan string
    resultChan  chan Result
    wg          sync.WaitGroup
}

// NewProcessor 创建新的OTP处理器
func NewProcessor(modelPath string, workerCount int) (*Processor, error) {
    det, err := detector.NewBalancedEnhancedOTPDetector(modelPath)
    if err != nil {
        return nil, err
    }
    
    p := &Processor{
        detector:    det,
        workerCount: workerCount,
        jobChan:     make(chan string, 1000),
        resultChan:  make(chan Result, 1000),
    }
    
    // 启动工作协程
    for i := 0; i < workerCount; i++ {
        p.wg.Add(1)
        go p.worker()
    }
    
    return p, nil
}

// worker 工作协程
func (p *Processor) worker() {
    defer p.wg.Done()
    
    for message := range p.jobChan {
        score, _ := p.detector.GetOTPScore(message)
        isOTP := score >= p.detector.GetDecisionThreshold()
        
        p.resultChan <- Result{
            Message: message,
            IsOTP:   isOTP,
            Score:   score,
        }
    }
}

// Process 处理单条消息
func (p *Processor) Process(message string) {
    p.jobChan <- message
}

// GetResult 获取处理结果
func (p *Processor) GetResult() Result {
    return <-p.resultChan
}

// Close 关闭处理器
func (p *Processor) Close() {
    close(p.jobChan)
    p.wg.Wait()
    close(p.resultChan)
}
```

使用示例：

```go
processor, err := otpprocessor.NewProcessor("models/go_params/otp_svm_balanced_enhanced_params.json", 10)
if err != nil {
    log.Fatal(err)
}
defer processor.Close()

// 处理消息
for _, message := range messages {
    processor.Process(message)
}

// 获取结果
for i := 0; i < len(messages); i++ {
    result := processor.GetResult()
    // 处理结果...
}
```

### 4.5 处理语言检测错误

当前模型可能在某些语言检测上存在错误，例如爱沙尼亚语有时会被误识别为英语。在实际产品中，可以通过以下方式处理：

```go
// 增强语言检测
func enhanceLanguageDetection(message string) string {
    // 爱沙尼亚语特有字符
    estonianChars := "õäöüÕÄÖÜ"
    
    // 检查是否包含爱沙尼亚语特有字符
    for _, char := range message {
        if strings.ContainsRune(estonianChars, char) {
            return "et" // 爱沙尼亚语
        }
    }
    
    // 使用原始语言检测
    detector := detector.NewLanguageDetector()
    return detector.DetectLanguage(message)
}
```

## 5. 性能优化

- **内存优化**：检测器在初始化时会加载整个模型到内存中（约2-3MB），使用单例模式避免重复加载
- **CPU优化**：每次调用`IsOTP`方法都会进行文本预处理和特征提取，可以考虑使用结果缓存
- **并发优化**：使用工作池模式处理大量消息，避免创建过多goroutine
- **批处理**：对于大量消息，使用批处理而非逐条处理
- **预热**：在服务启动时，使用一些示例消息预热模型，避免冷启动延迟

### 5.1 结果缓存

对于重复出现的消息，可以使用缓存避免重复计算：

```go
type CachedDetector struct {
    detector *detector.BalancedEnhancedOTPDetector
    cache    map[string]bool
    mu       sync.RWMutex
}

func NewCachedDetector(modelPath string) (*CachedDetector, error) {
    det, err := detector.NewBalancedEnhancedOTPDetector(modelPath)
    if err != nil {
        return nil, err
    }
    
    return &CachedDetector{
        detector: det,
        cache:    make(map[string]bool),
    }, nil
}

func (c *CachedDetector) IsOTP(message string) bool {
    // 先检查缓存
    c.mu.RLock()
    if result, ok := c.cache[message]; ok {
        c.mu.RUnlock()
        return result
    }
    c.mu.RUnlock()
    
    // 缓存未命中，执行检测
    result := c.detector.IsOTP(message)
    
    // 更新缓存
    c.mu.Lock()
    c.cache[message] = result
    c.mu.Unlock()
    
    return result
}
```

## 6. 故障排除

如果遇到问题，请检查：

1. 模型参数文件路径是否正确
2. JSON文件格式是否有效
3. 启用调试模式，查看更多信息：`detector.EnableDebug(true)`
4. 确保模型类型与参数文件匹配（svm、nb或rf）
5. 检查是否有足够的内存加载模型
6. 对于中文文本，确保已正确处理字符编码
7. 检查语言检测是否正确，特别是对于爱沙尼亚语等小语种

### 6.1 常见错误及解决方案

1. **模型加载失败**：
   ```
   读取模型参数文件失败: open models/go_params/otp_svm_balanced_enhanced_params.json: no such file or directory
   ```
   解决方案：检查文件路径，确保模型文件存在且有读取权限

2. **JSON解析错误**：
   ```
   解析模型参数失败: invalid character '}' after object key
   ```
   解决方案：检查JSON文件格式，确保格式正确

3. **内存不足**：
   ```
   runtime: out of memory
   ```
   解决方案：增加应用程序的内存限制，或使用更小的模型

4. **语言检测错误**：
   ```
   LANG_en (应为LANG_et)
   ```
   解决方案：使用增强的语言检测方法，或在处理特定语言时添加额外检查

## 7. 模型性能比较

| 模型 | 平衡方法 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|---------|-------|-------|-------|-------|-----|
| **SVM**  | **balanced_enhanced** | **0.90** | **0.86** | **0.83** | **0.8539** | **0.89** |
| NB   | balanced_enhanced | 0.90 | 0.88 | 0.81 | 0.8433 | 0.91 |
| RF   | balanced_enhanced | 0.88 | 0.81 | 0.84 | 0.8421 | 0.90 |
| NB   | smote | 0.9014 | 0.9140 | 0.7735 | 0.8379 | 0.9139 |
| SVM  | undersample | 0.8002 | 0.6412 | 0.8934 | 0.7465 | 0.8853 |
| SVM  | smote | 0.7643 | 0.6072 | 0.8049 | 0.6922 | 0.8420 |
| RF   | smote | 0.7785 | 0.9053 | 0.3657 | 0.5210 | 0.9183 |

**推荐模型**：SVM + balanced_enhanced (F1分数最高)

## 8. 多语言支持

SVM+balanced_enhanced模型支持以下语言的OTP检测：

- 英文 (en)
- 中文 (zh)
- 俄语 (ru)
- 爱沙尼亚语 (et)

注意：爱沙尼亚语检测可能不够准确，有时会被误识别为英语，但这通常不会影响OTP分类结果。 