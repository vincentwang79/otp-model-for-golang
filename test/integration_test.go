package test

import (
	"fmt"
	"testing"

	"../detector"
)

// 测试词嵌入检测器
func TestWordEmbeddingDetector(t *testing.T) {
	// 创建词嵌入检测器
	modelPath := "../models/word_embedding_svm_model.joblib"
	thresholdPath := "../models/word_embedding_svm_threshold.txt"
	embeddingPath := "../models/word_embedding_vectors.json"

	detector, err := detector.NewWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath, false)
	if err != nil {
		t.Fatalf("创建词嵌入检测器失败: %v", err)
	}

	// 测试用例
	testCases := []struct {
		text     string
		expected bool
	}{
		{"Your verification code is 123456", true},
		{"Use code 987654 to login", true},
		{"Your OTP is 246810", true},
		{"Hello, how are you today?", false},
		{"Meeting at 5pm tomorrow", false},
		{"您的验证码是135790，请勿泄露", true},
		{"今天天气真好，我们去公园吧", false},
		{"Ваш код подтверждения: 112233", true},
		{"Хорошего дня!", false},
		{"Teie kinnituskood on 445566", true},
		{"Tere hommikust", false},
	}

	// 运行测试
	for i, tc := range testCases {
		result := detector.IsOTP(tc.text)
		if result != tc.expected {
			t.Errorf("测试用例 %d 失败: 文本 %q, 期望 %v, 得到 %v", i+1, tc.text, tc.expected, result)
		}
	}
}

// 测试词嵌入检测器与现有检测器的集成
func TestDetectorIntegration(t *testing.T) {
	// 创建词嵌入检测器
	modelPath := "../models/word_embedding_svm_model.joblib"
	thresholdPath := "../models/word_embedding_svm_threshold.txt"
	embeddingPath := "../models/word_embedding_vectors.json"

	wordEmbeddingDetector, err := detector.NewWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath, false)
	if err != nil {
		t.Fatalf("创建词嵌入检测器失败: %v", err)
	}

	// 创建现有检测器
	balancedDetector := detector.NewDetectorBalancedEnhanced()

	// 测试用例
	testCases := []struct {
		text string
	}{
		{"Your verification code is 123456"},
		{"Use code 987654 to login"},
		{"Your OTP is 246810"},
		{"Hello, how are you today?"},
		{"Meeting at 5pm tomorrow"},
		{"您的验证码是135790，请勿泄露"},
		{"今天天气真好，我们去公园吧"},
		{"Ваш код подтверждения: 112233"},
		{"Хорошего дня!"},
		{"Teie kinnituskood on 445566"},
		{"Tere hommikust"},
	}

	// 运行测试
	fmt.Println("词嵌入检测器与现有检测器的比较:")
	fmt.Println("----------------------------------------")
	fmt.Printf("%-40s | %-15s | %-15s\n", "文本", "词嵌入检测器", "现有检测器")
	fmt.Println("----------------------------------------")

	for _, tc := range testCases {
		wordEmbeddingResult := wordEmbeddingDetector.IsOTP(tc.text)
		balancedResult := balancedDetector.IsOTP(tc.text)

		wordEmbeddingStr := "非OTP短信"
		if wordEmbeddingResult {
			wordEmbeddingStr = "OTP短信"
		}

		balancedStr := "非OTP短信"
		if balancedResult {
			balancedStr = "OTP短信"
		}

		fmt.Printf("%-40s | %-15s | %-15s\n", truncateText(tc.text, 40), wordEmbeddingStr, balancedStr)
	}
}

// 截断文本
func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen-3] + "..."
}
