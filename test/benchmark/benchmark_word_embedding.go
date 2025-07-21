package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vincentwang79/otp-model-for-golang/detector"
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
	"Your one-time password is 654321. Please do not share this with anyone.",
	"Hi, can we meet at the coffee shop at 3pm?",
	"验证码：876543，5分钟内有效，请勿泄露给他人。",
	"下午好！请问你今天有空吗？",
	"Ваш пароль для входа: 112233. Действителен в течение 10 минут.",
	"Привет, как дела? Давно не виделись!",
	"Sisselogimise kood: 778899. Kood kehtib 5 minutit.",
	"Tere! Kuidas sul täna läheb?",
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

	// 等待用户按任意键退出
	fmt.Println("\n按回车键退出...")
	bufio.NewReader(os.Stdin).ReadString('\n')
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
