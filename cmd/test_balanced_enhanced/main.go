package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
	// 获取当前目录
	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Printf("无法获取当前目录: %v\n", err)
		return
	}

	// 模型参数文件路径
	modelPath := filepath.Join(currentDir, "models", "go_params", "otp_nb_smote_params.json")

	// 检查模型文件是否存在
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("模型文件不存在: %s\n", modelPath)
		return
	}

	// 创建平衡数据增强特征OTP检测器
	fmt.Printf("正在加载NB模型 (SMOTE平衡 + 增强特征)...\n")
	otpDetector, err := detector.NewBalancedEnhancedOTPDetector(modelPath)
	if err != nil {
		fmt.Printf("创建平衡数据增强特征OTP检测器失败: %v\n", err)
		return
	}

	// 启用调试模式
	otpDetector.EnableDebug(true)

	// 测试一些示例
	testExamples(otpDetector)
}

func testExamples(detector *detector.BalancedEnhancedOTPDetector) {
	examples := []struct {
		message string
		lang    string
	}{
		// 英文OTP示例
		{"Your verification code is 123456. Please enter it to complete the login.", "en"},
		{"Your OTP for bank transaction is 987654. Valid for 5 minutes.", "en"},

		// 英文非OTP示例
		{"Hi, how are you doing today? Let's meet for coffee tomorrow.", "en"},
		{"Meeting scheduled for tomorrow at 2pm. Please confirm your attendance.", "en"},

		// 中文OTP示例
		{"【云服务平台】验证码：135337，有效期为5分钟，请勿泄露。", "zh"},
		{"[ShopEase] 验证码：524304。请勿告诉他人。", "zh"},

		// 中文非OTP示例
		{"您好，您的订单已发货，预计明天送达，请保持电话畅通。", "zh"},
		{"感谢您的购买，我们将于明天安排送货上门。", "zh"},
	}

	// 处理示例
	for i, example := range examples {
		// 使用IsOTP方法直接判断
		isOTP := detector.IsOTP(example.message)

		// 同时获取得分和调试信息
		score, debugInfo := detector.GetOTPScore(example.message)

		fmt.Printf("\n示例 %d (%s):\n", i+1, example.lang)
		fmt.Printf("消息: %s\n", example.message)
		fmt.Printf("OTP得分: %.4f (阈值: %.4f)\n", score, debugInfo["decision_threshold"].(float64))
		fmt.Printf("检测结果: %v\n", isOTP)

		// 显示调试信息
		if debugInfo["processed_text"] != nil {
			fmt.Printf("处理后文本: %s\n", debugInfo["processed_text"])
		}
	}
}
