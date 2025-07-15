package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	detector "github.com/vincentwang79/otp-model-for-golang/src/go/detector"
)

func main() {
	// 解析命令行参数
	var (
		modelPath     string
		modelType     string
		balanceMethod string
		interactive   bool
		debug         bool
		benchmark     bool
		messageFile   string
	)

	flag.StringVar(&modelPath, "model", "", "模型参数文件路径")
	flag.StringVar(&modelType, "type", "nb", "模型类型: svm, nb (默认: nb)")
	flag.StringVar(&balanceMethod, "balance", "smote", "平衡方法: smote, undersample (默认: smote)")
	flag.BoolVar(&interactive, "interactive", false, "交互模式")
	flag.BoolVar(&debug, "debug", false, "调试模式")
	flag.BoolVar(&benchmark, "benchmark", false, "基准测试模式")
	flag.StringVar(&messageFile, "file", "", "消息文件路径")
	flag.Parse()

	// 如果未指定模型路径，使用默认模型参数
	if modelPath == "" {
		// 获取当前目录
		currentDir, err := os.Getwd()
		if err != nil {
			log.Fatalf("无法获取当前目录: %v", err)
		}

		// 根据模型类型和平衡方法选择模型参数文件
		modelPath = filepath.Join(currentDir, "..", "..", "models", "go_params",
			fmt.Sprintf("otp_%s_%s_params.json", modelType, balanceMethod))
	}

	// 检查模型文件是否存在
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("模型文件不存在: %s", modelPath)
	}

	// 创建OTP检测器
	fmt.Printf("正在加载%s模型 (%s平衡)...\n", modelType, balanceMethod)
	otpDetector, err := detector.NewImprovedOTPDetector(modelPath)
	if err != nil {
		log.Fatalf("创建OTP检测器失败: %v", err)
	}

	// 设置调试模式
	otpDetector.EnableDebug(debug)

	// 根据模式执行不同的操作
	if benchmark {
		runBenchmark(otpDetector)
	} else if messageFile != "" {
		processMessageFile(otpDetector, messageFile)
	} else if interactive {
		runInteractive(otpDetector)
	} else {
		// 默认运行示例
		runExamples(otpDetector)
	}
}

// 运行示例
func runExamples(detector *detector.ImprovedOTPDetector) {
	fmt.Println("===== 示例消息测试 (改进模型) =====")

	examples := []struct {
		Message  string
		Expected bool
	}{
		{"Your verification code is 123456. Please enter it to complete the login.", true},
		{"Your OTP for bank transaction is 987654. Valid for 5 minutes.", true},
		{"Hi, how are you doing today? Let's meet for coffee tomorrow.", false},
		{"Congratulations! You've won a free gift. Claim using code 123456.", false},
		{"Your account verification code: 654321. Do not share with anyone.", true},
		{"Meeting scheduled for tomorrow at 2pm. Please confirm your attendance.", false},
		{"Your one-time password is 246810. Use it within 10 minutes.", true},
		{"Special offer: 50% discount on all items. Limited time only!", false},
		{"Your security code for password reset is 135790. Valid for 15 minutes.", true},
		{"Thank you for your purchase. Your order will be delivered tomorrow.", false},
		{"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。", true},
		{"[ShopEase] 验证码：524304。请勿告诉他人。", true},
		{"您好，您的订单已发货，预计明天送达，请保持电话畅通。", false},
		{"欢迎使用[极速租车]，您的验证码为：855760。", true},
	}

	correct := 0
	for i, example := range examples {
		isOTP, confidence, err := detector.IsOTP(example.Message)
		if err != nil {
			log.Printf("消息 #%d 检测失败: %v", i+1, err)
			continue
		}

		result := "✓ 正确"
		if isOTP != example.Expected {
			result = "✗ 错误"
		} else {
			correct++
		}

		fmt.Printf("消息 #%d: %s\n", i+1, example.Message)
		fmt.Printf("预期: %v, 实际: %v, 置信度: %.4f - %s\n\n", example.Expected, isOTP, confidence, result)
	}

	fmt.Printf("准确率: %.2f%% (%d/%d)\n", float64(correct)/float64(len(examples))*100, correct, len(examples))
}

// 交互模式
func runInteractive(detector *detector.ImprovedOTPDetector) {
	fmt.Println("===== 交互模式 (改进模型) =====")
	fmt.Println("输入消息进行OTP检测，输入'quit'或'exit'退出")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		message := scanner.Text()
		if message == "quit" || message == "exit" {
			break
		}

		if message == "" {
			continue
		}

		isOTP, confidence, err := detector.IsOTP(message)
		if err != nil {
			fmt.Printf("检测失败: %v\n", err)
			continue
		}

		fmt.Printf("是否为OTP: %v, 置信度: %.4f\n", isOTP, confidence)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("读取输入失败: %v\n", err)
	}
}

// 处理消息文件
func processMessageFile(detector *detector.ImprovedOTPDetector, filePath string) {
	fmt.Printf("===== 处理文件 (改进模型): %s =====\n", filePath)

	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("无法打开文件: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var totalCount, otpCount int

	// 处理每一行
	for scanner.Scan() {
		message := scanner.Text()
		if message == "" {
			continue
		}

		totalCount++

		isOTP, _, err := detector.IsOTP(message)
		if err != nil {
			fmt.Printf("检测失败: %v\n", err)
			continue
		}

		if isOTP {
			otpCount++
		}

		// 每处理100条消息显示一次进度
		if totalCount%100 == 0 {
			fmt.Printf("已处理: %d 条消息\n", totalCount)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("读取文件失败: %v", err)
	}

	fmt.Printf("\n总消息数: %d\n", totalCount)
	fmt.Printf("OTP消息数: %d (%.2f%%)\n", otpCount, float64(otpCount)/float64(totalCount)*100)
	fmt.Printf("非OTP消息数: %d (%.2f%%)\n", totalCount-otpCount, float64(totalCount-otpCount)/float64(totalCount)*100)
}

// 运行基准测试
func runBenchmark(detector *detector.ImprovedOTPDetector) {
	fmt.Println("===== 基准测试 (改进模型) =====")

	// 准备测试数据
	messages := []string{
		"Your verification code is 123456. Please enter it to complete the login.",
		"Your OTP for bank transaction is 987654. Valid for 5 minutes.",
		"Hi, how are you doing today? Let's meet for coffee tomorrow.",
		"Congratulations! You've won a free gift. Claim using code 123456.",
		"Your account verification code: 654321. Do not share with anyone.",
		"Meeting scheduled for tomorrow at 2pm. Please confirm your attendance.",
		"Your one-time password is 246810. Use it within 10 minutes.",
		"Special offer: 50% discount on all items. Limited time only!",
		"Your security code for password reset is 135790. Valid for 15 minutes.",
		"Thank you for your purchase. Your order will be delivered tomorrow.",
		"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。",
		"[ShopEase] 验证码：524304。请勿告诉他人。",
		"您好，您的订单已发货，预计明天送达，请保持电话畅通。",
		"欢迎使用[极速租车]，您的验证码为：855760。",
	}

	// 重复消息以创建更大的测试集
	var testMessages []string
	for i := 0; i < 1000; i++ {
		testMessages = append(testMessages, messages...)
	}

	fmt.Printf("测试消息数: %d\n", len(testMessages))

	// 开始计时
	start := time.Now()

	// 处理消息
	for _, message := range testMessages {
		_, _, err := detector.IsOTP(message)
		if err != nil {
			log.Fatalf("检测失败: %v", err)
		}
	}

	// 计算耗时和性能
	duration := time.Since(start)
	messagesPerSecond := float64(len(testMessages)) / duration.Seconds()

	fmt.Printf("总耗时: %v\n", duration)
	fmt.Printf("每秒处理消息数: %.2f\n", messagesPerSecond)
	fmt.Printf("每条消息平均耗时: %.2f µs\n", float64(duration.Nanoseconds())/float64(len(testMessages))/1000)
}
