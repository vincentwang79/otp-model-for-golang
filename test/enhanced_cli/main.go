package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	detector "github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
	// 解析命令行参数
	var (
		modelPath   string
		modelType   string
		interactive bool
		debug       bool
		benchmark   bool
		messageFile string
	)

	flag.StringVar(&modelPath, "model", "", "模型参数文件路径")
	flag.StringVar(&modelType, "type", "nb", "模型类型: svm, nb, rf (默认: nb)")
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

		// 根据模型类型选择模型参数文件
		modelPath = filepath.Join(currentDir, "..", "..", "models", "go_params",
			fmt.Sprintf("otp_%s_enhanced_params.json", modelType))
	}

	// 检查模型文件是否存在
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("模型文件不存在: %s", modelPath)
	}

	// 创建增强OTP检测器
	fmt.Printf("正在加载增强特征工程的%s模型...\n", modelType)
	otpDetector, err := detector.NewEnhancedOTPDetector(modelPath)
	if err != nil {
		log.Fatalf("创建增强OTP检测器失败: %v", err)
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
func runExamples(detector *detector.EnhancedOTPDetector) {
	fmt.Println("===== 示例消息测试 (增强特征工程) =====")

	examples := []struct {
		Message  string
		Expected bool
		Language string
	}{
		{"Your verification code is 123456. Please enter it to complete the login.", true, "英文"},
		{"Your OTP for bank transaction is 987654. Valid for 5 minutes.", true, "英文"},
		{"Hi, how are you doing today? Let's meet for coffee tomorrow.", false, "英文"},
		{"Congratulations! You've won a free gift. Claim using code 123456.", false, "英文"},
		{"Your account verification code: 654321. Do not share with anyone.", true, "英文"},
		{"Meeting scheduled for tomorrow at 2pm. Please confirm your attendance.", false, "英文"},
		{"Your one-time password is 246810. Use it within 10 minutes.", true, "英文"},
		{"Special offer: 50% discount on all items. Limited time only!", false, "英文"},
		{"Your security code for password reset is 135790. Valid for 15 minutes.", true, "英文"},
		{"Thank you for your purchase. Your order will be delivered tomorrow.", false, "英文"},
		{"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。", true, "中文"},
		{"[ShopEase] 验证码：524304。请勿告诉他人。", true, "中文"},
		{"您好，您的订单已发货，预计明天送达，请保持电话畅通。", false, "中文"},
		{"欢迎使用[极速租车]，您的验证码为：855760。", true, "中文"},
		{"Ваш проверочный код: 123456. Действителен в течение 5 минут.", true, "俄语"},
		{"Одноразовый пароль для входа в систему: 987654", true, "俄语"},
		{"Здравствуйте! Ваш заказ успешно оформлен и будет доставлен завтра.", false, "俄语"},
		{"Специальное предложение: скидка 50% на все товары!", false, "俄语"},
		{"Teie kinnituskood on 123456. Kehtib 5 minutit.", true, "爱沙尼亚语"},
		{"Sisselogimiseks kasutage ühekordset parooli: 654321", true, "爱沙尼亚语"},
		{"Tere! Teie tellimus on edukalt vormistatud ja toimetatakse kohale homme.", false, "爱沙尼亚语"},
		{"Eripakkumine: 50% allahindlus kõikidele toodetele!", false, "爱沙尼亚语"},
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

		fmt.Printf("消息 #%d (%s): %s\n", i+1, example.Language, example.Message)
		fmt.Printf("预期: %v, 实际: %v, 置信度: %.4f - %s\n\n", example.Expected, isOTP, confidence, result)
	}

	fmt.Printf("准确率: %.2f%% (%d/%d)\n", float64(correct)/float64(len(examples))*100, correct, len(examples))
}

// 交互模式
func runInteractive(detector *detector.EnhancedOTPDetector) {
	fmt.Println("===== 交互模式 (增强特征工程) =====")
	fmt.Println("输入消息文本，按回车键检测，输入'exit'退出")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		text := scanner.Text()
		if text == "exit" {
			break
		}

		if text == "" {
			continue
		}

		isOTP, confidence, err := detector.IsOTP(text)
		if err != nil {
			fmt.Printf("检测失败: %v\n", err)
			continue
		}

		fmt.Printf("是否为OTP短信: %v, 置信度: %.4f\n", isOTP, confidence)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "读取输入失败: %v\n", err)
	}
}

// 处理消息文件
func processMessageFile(detector *detector.EnhancedOTPDetector, filePath string) {
	fmt.Printf("===== 处理文件: %s =====\n", filePath)

	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("无法打开文件: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineNum := 0
	otpCount := 0

	// 处理每一行
	for scanner.Scan() {
		lineNum++
		message := scanner.Text()
		if message == "" {
			continue
		}

		isOTP, confidence, err := detector.IsOTP(message)
		if err != nil {
			fmt.Printf("行 %d 检测失败: %v\n", lineNum, err)
			continue
		}

		if isOTP {
			otpCount++
		}

		fmt.Printf("行 %d: %s\n", lineNum, message)
		fmt.Printf("是否为OTP短信: %v, 置信度: %.4f\n\n", isOTP, confidence)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("读取文件失败: %v", err)
	}

	fmt.Printf("处理完成，共 %d 行，检测到 %d 条OTP短信\n", lineNum, otpCount)
}

// 基准测试
func runBenchmark(detector *detector.EnhancedOTPDetector) {
	fmt.Println("===== 基准测试 (增强特征工程) =====")

	// 示例消息列表
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
		"Ваш проверочный код: 123456. Действителен в течение 5 минут.",
		"Одноразовый пароль для входа в систему: 987654",
		"Здравствуйте! Ваш заказ успешно оформлен и будет доставлен завтра.",
		"Специальное предложение: скидка 50% на все товары!",
		"Teie kinnituskood on 123456. Kehtib 5 minutit.",
		"Sisselogimiseks kasutage ühekordset parooli: 654321",
	}

	// 重复消息以增加测试量
	var allMessages []string
	for i := 0; i < 5000; i++ {
		allMessages = append(allMessages, messages...)
	}

	fmt.Printf("准备测试 %d 条消息...\n", len(allMessages))

	// 开始计时
	start := time.Now()

	// 处理所有消息
	otpCount := 0
	for _, message := range allMessages {
		isOTP, _, err := detector.IsOTP(message)
		if err != nil {
			log.Printf("检测失败: %v", err)
			continue
		}
		if isOTP {
			otpCount++
		}
	}

	// 计算耗时
	duration := time.Since(start)
	messagesPerSecond := float64(len(allMessages)) / duration.Seconds()

	fmt.Printf("共处理 %d 条消息，检测到 %d 条OTP短信\n", len(allMessages), otpCount)
	fmt.Printf("总耗时: %v\n", duration)
	fmt.Printf("每秒处理消息数: %.2f\n", messagesPerSecond)
	fmt.Printf("每条消息平均处理时间: %.2f 微秒\n", float64(duration.Nanoseconds())/float64(len(allMessages))/1000)
}
