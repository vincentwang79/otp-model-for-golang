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
		modelPath     string
		modelType     string
		balanceMethod string
		interactive   bool
		debug         bool
		benchmark     bool
		messageFile   string
	)

	flag.StringVar(&modelPath, "model", "", "模型参数文件路径")
	flag.StringVar(&modelType, "type", "nb", "模型类型: svm, nb, rf (默认: nb)")
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

	// 创建平衡数据增强特征OTP检测器
	fmt.Printf("正在加载%s模型 (%s平衡 + 增强特征)...\n", modelType, balanceMethod)
	otpDetector, err := detector.NewBalancedEnhancedOTPDetector(modelPath)
	if err != nil {
		log.Fatalf("创建平衡数据增强特征OTP检测器失败: %v", err)
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
func runExamples(detector *detector.BalancedEnhancedOTPDetector) {
	fmt.Println("===== 示例测试 (平衡数据 + 增强特征) =====")

	examples := []struct {
		message string
		lang    string
	}{
		// 英文OTP示例
		{"Your verification code is 123456. Please enter it to complete the login.", "en"},
		{"Your OTP for bank transaction is 987654. Valid for 5 minutes.", "en"},
		{"Your one-time password is 246810. Use it within 10 minutes.", "en"},
		{"Your security code for password reset is 135790. Valid for 15 minutes.", "en"},

		// 英文非OTP示例
		{"Hi, how are you doing today? Let's meet for coffee tomorrow.", "en"},
		{"Meeting scheduled for tomorrow at 2pm. Please confirm your attendance.", "en"},
		{"Special offer: 50% discount on all items. Limited time only!", "en"},
		{"Thank you for your purchase. Your order will be delivered tomorrow.", "en"},

		// 中文OTP示例
		{"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。", "zh"},
		{"[ShopEase] 验证码：524304。请勿告诉他人。", "zh"},
		{"欢迎使用[极速租车]，您的验证码为：855760。", "zh"},
		{"您的登录验证码是123456，10分钟内有效，请勿泄露。", "zh"},

		// 中文非OTP示例
		{"您好，您的订单已发货，预计明天送达，请保持电话畅通。", "zh"},
		{"感谢您的购买，我们将于明天安排送货上门。", "zh"},
		{"特别优惠：所有商品五折起，限时抢购！", "zh"},
		{"您的会员积分已更新，当前可用积分为1000分。", "zh"},

		// 俄语OTP示例
		{"Ваш проверочный код: 123456. Действителен в течение 5 минут.", "ru"},
		{"Одноразовый пароль для входа в систему: 987654", "ru"},

		// 俄语非OTP示例
		{"Здравствуйте! Ваш заказ успешно оформлен и будет доставлен завтра.", "ru"},
		{"Специальное предложение: скидка 50% на все товары!", "ru"},

		// 爱沙尼亚语OTP示例
		{"Teie kinnituskood on 123456. Kehtib 5 minutit.", "et"},
		{"Sisselogimiseks kasutage ühekordset parooli: 654321", "et"},

		// 爱沙尼亚语非OTP示例
		{"Tere! Teie tellimus on edukalt vormistatud ja toimetatakse kohale homme.", "et"},
		{"Eripakkumine: 50% allahindlus kõikidele toodetele!", "et"},
	}

	// 处理示例
	for i, example := range examples {
		start := time.Now()
		score, debugInfo := detector.GetOTPScore(example.message)
		isOTP := score >= debugInfo["decision_threshold"].(float64)
		duration := time.Since(start)

		fmt.Printf("\n示例 %d (%s):\n", i+1, example.lang)
		fmt.Printf("消息: %s\n", example.message)
		fmt.Printf("OTP得分: %.4f (阈值: %.4f)\n", score, debugInfo["decision_threshold"].(float64))
		fmt.Printf("检测结果: %v\n", isOTP)
		fmt.Printf("处理时间: %v\n", duration)

		// 如果开启调试模式，显示更多信息
		if debugInfo["processed_text"] != nil {
			fmt.Printf("处理后文本: %s\n", debugInfo["processed_text"])
		}
	}
}

// 交互模式
func runInteractive(detector *detector.BalancedEnhancedOTPDetector) {
	fmt.Println("===== 交互模式 (平衡数据 + 增强特征) =====")
	fmt.Println("输入短信内容，按Enter键检测是否为OTP短信")
	fmt.Println("输入'exit'或'quit'退出")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n请输入短信内容: ")
		if !scanner.Scan() {
			break
		}

		input := scanner.Text()
		if input == "exit" || input == "quit" {
			break
		}

		if input == "" {
			continue
		}

		start := time.Now()
		score, debugInfo := detector.GetOTPScore(input)
		isOTP := score >= debugInfo["decision_threshold"].(float64)
		duration := time.Since(start)

		fmt.Printf("OTP得分: %.4f (阈值: %.4f)\n", score, debugInfo["decision_threshold"].(float64))
		fmt.Printf("检测结果: %v\n", isOTP)
		fmt.Printf("处理时间: %v\n", duration)

		// 如果开启调试模式，显示更多信息
		if debugInfo["processed_text"] != nil {
			fmt.Printf("处理后文本: %s\n", debugInfo["processed_text"])
		}
	}
}

// 处理消息文件
func processMessageFile(detector *detector.BalancedEnhancedOTPDetector, filePath string) {
	fmt.Printf("===== 处理文件: %s (平衡数据 + 增强特征) =====\n", filePath)

	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("无法打开文件: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineNum := 0
	otpCount := 0

	start := time.Now()

	// 逐行处理
	for scanner.Scan() {
		lineNum++
		message := scanner.Text()

		if message == "" {
			continue
		}

		isOTP := detector.IsOTP(message)
		if isOTP {
			otpCount++
		}

		fmt.Printf("行 %d: %s -> %v\n", lineNum, message, isOTP)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("读取文件错误: %v", err)
	}

	duration := time.Since(start)
	fmt.Printf("\n处理完成: 总行数 %d, OTP短信 %d, 非OTP短信 %d\n", lineNum, otpCount, lineNum-otpCount)
	fmt.Printf("总处理时间: %v, 平均每条: %v\n", duration, duration/time.Duration(lineNum))
}

// 基准测试
func runBenchmark(detector *detector.BalancedEnhancedOTPDetector) {
	fmt.Println("===== 基准测试 (平衡数据 + 增强特征) =====")

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
	for i := 0; i < 500; i++ {
		allMessages = append(allMessages, messages...)
	}

	fmt.Printf("准备测试 %d 条消息...\n", len(allMessages))

	// 开始计时
	start := time.Now()

	// 处理所有消息
	otpCount := 0
	for _, msg := range allMessages {
		if detector.IsOTP(msg) {
			otpCount++
		}
	}

	// 计算耗时
	duration := time.Since(start)
	msPerMessage := float64(duration.Milliseconds()) / float64(len(allMessages))

	fmt.Printf("处理完成: 总消息数 %d, 检测为OTP %d, 非OTP %d\n",
		len(allMessages), otpCount, len(allMessages)-otpCount)
	fmt.Printf("总处理时间: %v, 平均每条: %.2f ms\n", duration, msPerMessage)
}
