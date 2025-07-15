package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	otpdetector "github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
	// 获取当前目录
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("无法获取当前目录: %v", err)
	}

	// 模型参数文件路径
	paramsPath := filepath.Join(currentDir, "..", "..", "models", "go_params", "otp_svm_params.json")

	// 数据集文件路径
	otpMessagesPath := filepath.Join(currentDir, "..", "..", "data", "otp_messages_1000.txt")
	nonOtpMessagesPath := filepath.Join(currentDir, "..", "..", "data", "non_otp_messages_1000.txt")

	// 检查文件是否存在
	for _, path := range []string{paramsPath, otpMessagesPath, nonOtpMessagesPath} {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			log.Fatalf("文件不存在: %s", path)
		}
	}

	// 创建OTP检测器
	fmt.Println("正在加载OTP检测模型...")
	otpDetector, err := otpdetector.NewOTPDetector(paramsPath)
	if err != nil {
		log.Fatalf("创建OTP检测器失败: %v", err)
	}

	// 禁用调试模式以获得更好的性能
	otpDetector.EnableDebug(false)

	// 验证OTP消息
	fmt.Println("\n===== 验证OTP消息 =====")
	otpResults := validateDataset(otpDetector, otpMessagesPath, true)

	// 验证非OTP消息
	fmt.Println("\n===== 验证非OTP消息 =====")
	nonOtpResults := validateDataset(otpDetector, nonOtpMessagesPath, false)

	// 计算总体准确率
	totalMessages := otpResults.totalCount + nonOtpResults.totalCount
	correctPredictions := otpResults.correctCount + nonOtpResults.correctCount
	accuracy := float64(correctPredictions) / float64(totalMessages) * 100

	fmt.Println("\n===== 总体结果 =====")
	fmt.Printf("总消息数: %d\n", totalMessages)
	fmt.Printf("正确预测数: %d\n", correctPredictions)
	fmt.Printf("准确率: %.2f%%\n", accuracy)
	fmt.Printf("总耗时: %v\n", otpResults.duration+nonOtpResults.duration)
	fmt.Printf("每秒处理消息数: %.2f\n", float64(totalMessages)/
		(otpResults.duration.Seconds()+nonOtpResults.duration.Seconds()))

	// 打印详细的分类指标
	fmt.Println("\n===== 分类指标 =====")
	// 计算真阳性、假阳性、真阴性、假阴性
	truePositives := otpResults.correctCount
	falseNegatives := otpResults.totalCount - otpResults.correctCount
	trueNegatives := nonOtpResults.correctCount
	falsePositives := nonOtpResults.totalCount - nonOtpResults.correctCount

	// 计算精确率、召回率和F1分数
	precision := float64(truePositives) / float64(truePositives+falsePositives)
	recall := float64(truePositives) / float64(truePositives+falseNegatives)
	f1Score := 2 * precision * recall / (precision + recall)

	fmt.Printf("真阳性 (TP): %d\n", truePositives)
	fmt.Printf("假阳性 (FP): %d\n", falsePositives)
	fmt.Printf("真阴性 (TN): %d\n", trueNegatives)
	fmt.Printf("假阴性 (FN): %d\n", falseNegatives)
	fmt.Printf("精确率: %.4f\n", precision)
	fmt.Printf("召回率: %.4f\n", recall)
	fmt.Printf("F1分数: %.4f\n", f1Score)

	// 示例消息测试
	fmt.Println("\n===== 示例消息测试 =====")
	testExampleMessages(otpDetector)
}

type validationResult struct {
	totalCount   int
	correctCount int
	duration     time.Duration
}

// 验证数据集并返回结果
func validateDataset(otpDetector *otpdetector.OTPDetector, datasetPath string, expectedOTP bool) validationResult {
	// 打开数据集文件
	file, err := os.Open(datasetPath)
	if err != nil {
		log.Fatalf("无法打开数据集文件: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var totalCount, correctCount int
	var errors []string

	// 开始计时
	start := time.Now()

	// 处理每一行
	for scanner.Scan() {
		message := scanner.Text()
		if message == "" {
			continue
		}

		totalCount++

		// 检测是否为OTP
		isOTP, confidence, err := otpDetector.IsOTP(message)
		if err != nil {
			errors = append(errors, fmt.Sprintf("处理消息失败: %v", err))
			continue
		}

		// 检查预测是否正确
		if isOTP == expectedOTP {
			correctCount++
		} else {
			// 记录错误预测，但最多只记录10个
			if len(errors) < 10 {
				errors = append(errors, fmt.Sprintf(
					"预测错误 - 消息: %s\n预期: %v, 实际: %v, 置信度: %.4f",
					truncateString(message, 100), expectedOTP, isOTP, confidence))
			}
		}

		// 每处理100条消息显示一次进度
		if totalCount%100 == 0 {
			fmt.Printf("已处理: %d 条消息\n", totalCount)
		}
	}

	// 计算耗时
	duration := time.Since(start)

	// 打印结果
	expectedLabel := "OTP"
	if !expectedOTP {
		expectedLabel = "非OTP"
	}

	fmt.Printf("\n%s消息总数: %d\n", expectedLabel, totalCount)
	fmt.Printf("正确预测数: %d\n", correctCount)
	fmt.Printf("准确率: %.2f%%\n", float64(correctCount)/float64(totalCount)*100)
	fmt.Printf("处理耗时: %v\n", duration)
	fmt.Printf("每秒处理消息数: %.2f\n", float64(totalCount)/duration.Seconds())

	// 打印错误预测示例
	if len(errors) > 0 {
		fmt.Printf("\n错误预测示例 (最多10个):\n")
		for i, err := range errors {
			fmt.Printf("%d. %s\n\n", i+1, err)
		}
	}

	return validationResult{
		totalCount:   totalCount,
		correctCount: correctCount,
		duration:     duration,
	}
}

// 测试示例消息
func testExampleMessages(otpDetector *otpdetector.OTPDetector) {
	// 示例短信
	exampleMessages := []struct {
		Text     string
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
	}

	// 测试示例消息
	for i, example := range exampleMessages {
		isOTP, confidence, err := otpDetector.IsOTP(example.Text)
		if err != nil {
			log.Printf("消息 #%d 检测失败: %v", i+1, err)
			continue
		}

		result := "✓ 正确"
		if isOTP != example.Expected {
			result = "✗ 错误"
		}

		fmt.Printf("消息 #%d: %s\n", i+1, example.Text)
		fmt.Printf("预期: %v, 实际: %v, 置信度: %.4f - %s\n\n", example.Expected, isOTP, confidence, result)
	}
}

// 截断字符串，避免过长输出
func truncateString(s string, maxLength int) string {
	if len(s) <= maxLength {
		return s
	}
	return s[:maxLength] + "..."
}
