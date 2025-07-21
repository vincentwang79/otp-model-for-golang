# 词嵌入OTP检测器使用指南

## 概述

本文档介绍如何在项目中使用基于词嵌入(Word Embeddings)的OTP检测器。词嵌入检测器相比传统的TF-IDF模型有更好的语义理解能力和多语言支持。

## 安装与依赖

### Python部分

词嵌入模型的训练和评估需要以下Python依赖：

```bash
# 激活虚拟环境
source venv_py3/bin/activate

# 安装依赖
pip install gensim tensorflow transformers fasttext
pip install jieba pymorphy2 estnltk
```

### Go部分

Go实现不需要额外的依赖，但需要提前准备好以下文件：

1. 词嵌入向量文件 (`models/word_embedding_vectors.json`)
2. 模型阈值文件 (`models/word_embedding_svm_threshold.txt`)

## 使用方法

### 1. 使用Python训练词嵌入模型

```bash
# 评估不同词嵌入模型
python src/word_embeddings.py

# 使用词嵌入特征训练分类器
python src/word_embedding_model_trainer.py
```

这将在`models`目录下生成以下文件：
- `word_embedding_vectors.json`: 词嵌入向量
- `word_embedding_svm_model.joblib`: 训练好的SVM模型
- `word_embedding_svm_threshold.txt`: 最佳决策阈值

### 2. 在Go中使用词嵌入检测器

#### 作为独立检测器

```go
import (
    "fmt"
    "github.com/yourname/project/detector" // 根据实际项目修改导入路径
)

func main() {
    // 创建检测器
    modelPath := "models/word_embedding_svm_model.joblib"
    thresholdPath := "models/word_embedding_svm_threshold.txt"
    embeddingPath := "models/word_embedding_vectors.json"
    debug := false
    
    // 使用工厂方法创建检测器，避免命名冲突
    detector, err := detector.CreateWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath, debug)
    if err != nil {
        panic(err)
    }
    
    // 设置语言（可选，默认为"auto"）
    detector.SetLanguage("auto") // 可选值: "auto", "en", "zh", "ru", "et"
    
    // 检测短信
    text := "Your verification code is 123456"
    isOTP := detector.IsOTP(text)
    
    if isOTP {
        fmt.Println("这是一条OTP短信")
    } else {
        fmt.Println("这不是OTP短信")
    }
}
```

#### 与现有检测器集成

```go
import (
    "fmt"
    "github.com/yourname/project/detector" // 根据实际项目修改导入路径
)

func main() {
    // 创建词嵌入检测器
    wordEmbeddingDetector, err := detector.CreateWordEmbeddingDetector(
        "models/word_embedding_svm_model.joblib",
        "models/word_embedding_svm_threshold.txt",
        "models/word_embedding_vectors.json",
        false,
    )
    if err != nil {
        panic(err)
    }
    
    // 创建现有检测器
    balancedDetector := detector.NewDetectorBalancedEnhanced()
    
    // 检测短信
    text := "Your verification code is 123456"
    
    // 使用两个检测器进行检测
    isOTPByWordEmbedding := wordEmbeddingDetector.IsOTP(text)
    isOTPByBalanced := balancedDetector.IsOTP(text)
    
    // 融合结果（简单投票）
    isOTP := isOTPByWordEmbedding || isOTPByBalanced
    
    if isOTP {
        fmt.Println("这是一条OTP短信")
    } else {
        fmt.Println("这不是OTP短信")
    }
}
```

### 3. 创建和使用CLI工具

您可以创建一个命令行工具来测试词嵌入检测器。以下是实现和使用方法：

#### CLI工具实现示例

```go
package main

import (
    "bufio"
    "flag"
    "fmt"
    "os"
    "strings"

    "github.com/yourname/project/detector" // 根据实际项目修改
)

func main() {
    // 定义命令行参数
    modelPath := flag.String("model", "models/word_embedding_svm_model.joblib", "模型文件路径")
    thresholdPath := flag.String("threshold", "models/word_embedding_svm_threshold.txt", "阈值文件路径")
    embeddingPath := flag.String("embedding", "models/word_embedding_vectors.json", "词嵌入文件路径")
    language := flag.String("lang", "auto", "语言 (auto, en, zh, ru, et)")
    debug := flag.Bool("debug", false, "调试模式")
    interactive := flag.Bool("interactive", false, "交互模式")
    messageFlag := flag.String("message", "", "要检测的短信文本")

    flag.Parse()

    // 创建检测器
    detector, err := detector.CreateWordEmbeddingDetector(*modelPath, *thresholdPath, *embeddingPath, *debug)
    if err != nil {
        fmt.Printf("创建检测器失败: %v\n", err)
        os.Exit(1)
    }

    // 设置语言
    detector.SetLanguage(*language)

    // 交互模式
    if *interactive {
        runInteractiveMode(detector)
        return
    }

    // 单条消息模式
    if *messageFlag != "" {
        result := detector.IsOTP(*messageFlag)
        if result {
            fmt.Println("结果: OTP短信")
        } else {
            fmt.Println("结果: 非OTP短信")
        }
        return
    }

    // 如果没有指定交互模式或消息，显示帮助
    fmt.Println("请使用 -interactive 参数进入交互模式，或使用 -message 参数指定要检测的短信")
    flag.Usage()
}

// 运行交互模式
func runInteractiveMode(detector *detector.WordEmbeddingDetector) {
    scanner := bufio.NewScanner(os.Stdin)
    fmt.Println("词嵌入OTP检测器 - 交互模式")
    fmt.Println("输入短信内容进行检测，输入'exit'或'quit'退出")

    for {
        fmt.Print("> ")
        if !scanner.Scan() {
            break
        }

        text := scanner.Text()
        if text == "exit" || text == "quit" {
            break
        }

        if strings.TrimSpace(text) == "" {
            continue
        }

        result := detector.IsOTP(text)
        if result {
            fmt.Println("结果: OTP短信")
        } else {
            fmt.Println("结果: 非OTP短信")
        }
    }
}
```

#### 使用CLI工具

```bash
# 编译CLI工具
go build -o word_embedding_cli main.go

# 使用交互模式
./word_embedding_cli -interactive

# 检测单条短信
./word_embedding_cli -message "Your verification code is 123456"

# 指定语言
./word_embedding_cli -message "您的验证码是987654" -lang zh

# 启用调试模式
./word_embedding_cli -message "Your OTP is 246810" -debug
```

## 多语言支持

词嵌入检测器支持以下语言：

- 英语 (en)
- 中文 (zh)
- 俄语 (ru)
- 爱沙尼亚语 (et)

设置语言有两种方式：
1. 自动检测 (`auto`)
2. 手动指定 (`en`, `zh`, `ru`, `et`)

```go
detector.SetLanguage("zh") // 设置为中文
```

## 性能优化

词嵌入检测器的性能可以通过以下方式优化：

1. **减少词嵌入维度**：可以将词嵌入维度从300降低到100或50，以减少内存使用和计算量。

2. **限制词汇表大小**：只保留最常用的词汇，减少词嵌入向量文件的大小。

3. **使用量化技术**：对词嵌入向量进行量化，将浮点数转换为整数，减少内存使用。

4. **缓存常用词的向量**：对于常用的OTP关键词，可以预先计算并缓存其向量。

## 集成与性能

### 集成到现有项目

将词嵌入检测器集成到现有Go项目时，可能会遇到以下问题：

1. **导入路径**：根据项目结构，选择合适的导入方式：
   ```go
   // 同一项目内导入
   import "./detector"  // 简单项目结构
   
   // 或使用Go模块方式导入
   import "github.com/yourname/project/detector"  // 推荐用于正式项目
   ```

2. **Go模块配置**：如果使用Go模块，确保在go.mod中正确配置：
   ```
   module github.com/yourname/project
   
   go 1.16
   
   // 如果需要本地开发
   replace github.com/yourname/project/detector => ./detector
   ```

3. **函数命名冲突**：如果项目中有多个检测器实现，使用工厂方法避免命名冲突：
   ```go
   // 使用工厂函数创建检测器
   detector, err := detector.CreateWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath, debug)
   ```

### 性能基准

根据基准测试([详细结果](word_embedding_benchmark_results.md))，词嵌入检测器的性能表现如下：

- **处理速度**：每秒约25万条消息
- **性能提升**：比现有平衡增强检测器快约2.6倍
- **内存效率**：使用预计算的嵌入向量，减少运行时计算

## 故障排除

### 常见问题

1. **找不到模型文件**：确保`models`目录下有正确的词嵌入向量文件和阈值文件。

2. **内存使用过高**：考虑减少词嵌入维度或限制词汇表大小。

3. **检测结果不准确**：尝试调整决策阈值或重新训练模型。

4. **导入错误**：检查导入路径和go.mod配置，确保模块路径正确。

### 调试模式

启用调试模式可以查看更多信息：

```go
detector.SetDebug(true)
```

或在CLI中使用`-debug`参数：

```bash
./word_embedding_cli -message "Your code is 123456" -debug
```

## 参考资料

- [词嵌入模型性能评估](word_embeddings_performance.md)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [FastText: Library for efficient text classification and representation learning](https://fasttext.cc/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 

## 性能基准测试

如果您需要评估词嵌入检测器的性能，可以使用以下基准测试代码：

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
    "time"

    "github.com/yourname/project/detector" // 根据实际项目修改
)

// 测试数据
var testMessages = []string{
    "Your verification code is 123456",
    "Use code 987654 to login",
    "Your OTP is 246810",
    "Hello, how are you today?",
    "Meeting at 5pm tomorrow",
    "您的验证码是135790，请勿泄露",
    "今天天气真好，我们去公园吧",
    "Ваш код подтверждения: 112233",
    "Хорошего дня!",
    "Teie kinnituskood on 445566",
    "Tere hommikust",
    // 添加更多测试消息...
}

// 基准测试配置
const (
    warmupIterations  = 100
    benchmarkRuns     = 5
    iterationsPerRun  = 1000
    printProgressStep = 200
)

func main() {
    // 创建词嵌入检测器
    modelPath := "models/word_embedding_svm_model.joblib"
    thresholdPath := "models/word_embedding_svm_threshold.txt"
    embeddingPath := "models/word_embedding_vectors.json"

    fmt.Println("创建词嵌入检测器...")
    wordEmbeddingDetector, err := detector.CreateWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath, false)
    if err != nil {
        fmt.Printf("创建词嵌入检测器失败: %v\n", err)
        os.Exit(1)
    }

    // 创建现有检测器
    fmt.Println("创建平衡增强检测器...")
    balancedDetector := detector.NewDetectorBalancedEnhanced()

    // 预热
    fmt.Println("预热中...")
    for i := 0; i < warmupIterations; i++ {
        for _, msg := range testMessages {
            wordEmbeddingDetector.IsOTP(msg)
            balancedDetector.IsOTP(msg)
        }
    }

    // 基准测试词嵌入检测器
    fmt.Println("\n开始基准测试词嵌入检测器...")
    wordEmbeddingTimes := runBenchmark(wordEmbeddingDetector)

    // 基准测试平衡增强检测器
    fmt.Println("\n开始基准测试平衡增强检测器...")
    balancedTimes := runBenchmark(balancedDetector)

    // 输出结果
    printResults("词嵌入检测器", wordEmbeddingTimes)
    printResults("平衡增强检测器", balancedTimes)

    // 比较结果
    compareResults(wordEmbeddingTimes, balancedTimes)
}

// OTPDetector 接口定义检测器的共同方法
type OTPDetector interface {
    IsOTP(text string) bool
}

// 运行基准测试
func runBenchmark(detector OTPDetector) []float64 {
    times := make([]float64, benchmarkRuns)

    for run := 0; run < benchmarkRuns; run++ {
        fmt.Printf("运行 %d/%d...\n", run+1, benchmarkRuns)

        startTime := time.Now()
        for i := 0; i < iterationsPerRun; i++ {
            for _, msg := range testMessages {
                detector.IsOTP(msg)
            }

            if (i+1)%printProgressStep == 0 {
                fmt.Printf("  进度: %d/%d\n", i+1, iterationsPerRun)
            }
        }
        duration := time.Since(startTime).Seconds()
        times[run] = duration

        fmt.Printf("  完成! 用时: %.4f 秒\n", duration)
    }

    return times
}

// 打印结果
func printResults(name string, times []float64) {
    fmt.Printf("\n%s 性能结果:\n", name)
    fmt.Println(strings.Repeat("-", 40))

    var sum float64
    for i, t := range times {
        fmt.Printf("运行 %d: %.4f 秒\n", i+1, t)
        sum += t
    }

    avg := sum / float64(len(times))
    fmt.Printf("平均用时: %.4f 秒\n", avg)

    // 计算每秒处理的消息数
    messagesPerRun := iterationsPerRun * len(testMessages)
    messagesPerSecond := float64(messagesPerRun) / avg
    fmt.Printf("每秒处理消息数: %.2f\n", messagesPerSecond)
}

// 比较结果
func compareResults(wordEmbeddingTimes, balancedTimes []float64) {
    // 计算平均时间
    var weSum, bdSum float64
    for i := 0; i < benchmarkRuns; i++ {
        weSum += wordEmbeddingTimes[i]
        bdSum += balancedTimes[i]
    }
    weAvg := weSum / float64(benchmarkRuns)
    bdAvg := bdSum / float64(benchmarkRuns)

    // 计算每秒处理的消息数
    messagesPerRun := iterationsPerRun * len(testMessages)
    weMPS := float64(messagesPerRun) / weAvg
    bdMPS := float64(messagesPerRun) / bdAvg

    // 计算性能比较
    var faster string
    var speedup float64
    if weAvg < bdAvg {
        faster = "词嵌入检测器"
        speedup = bdAvg / weAvg
    } else {
        faster = "平衡增强检测器"
        speedup = weAvg / bdAvg
    }

    // 打印比较结果
    fmt.Println("\n性能比较:")
    fmt.Println(strings.Repeat("-", 40))
    fmt.Printf("词嵌入检测器平均用时: %.4f 秒 (%.2f 消息/秒)\n", weAvg, weMPS)
    fmt.Printf("平衡增强检测器平均用时: %.4f 秒 (%.2f 消息/秒)\n", bdAvg, bdMPS)
    fmt.Printf("%s 更快 %.2f 倍\n", faster, speedup)
}
```

运行此基准测试代码，您将获得类似于以下的结果：

```
词嵌入检测器平均用时: 0.0750 秒 (253,251.22 消息/秒)
平衡增强检测器平均用时: 0.1945 秒 (97,676.91 消息/秒)
词嵌入检测器 更快 2.59 倍
```

有关详细的基准测试结果和分析，请参阅[词嵌入基准测试结果](word_embedding_benchmark_results.md)。 