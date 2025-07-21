package detector

import (
	"fmt"
	"os"
	"path/filepath"
)

// 默认模型参数文件路径
const (
	DefaultModelPath                 = "models/otp_detector_model.json"
	DefaultImprovedModelPath         = "models/otp_detector_improved_model.json"
	DefaultBalancedEnhancedModelPath = "models/otp_detector_balanced_enhanced_model.json"
)

// NewDetector 创建一个基本的OTP检测器
func NewDetector() *OTPDetector {
	detector, err := NewOTPDetector(DefaultModelPath)
	if err != nil {
		fmt.Printf("警告: 无法加载OTP检测器模型: %v\n", err)
		fmt.Printf("使用默认参数初始化检测器\n")
		return &OTPDetector{
			params: ModelParams{
				Binary:    true,
				Lowercase: true,
				ModelType: "svm",
			},
			debug: false,
		}
	}
	return detector
}

// NewDetectorImproved 创建一个改进的OTP检测器
func NewDetectorImproved() *ImprovedOTPDetector {
	detector, err := NewImprovedOTPDetector(DefaultImprovedModelPath)
	if err != nil {
		fmt.Printf("警告: 无法加载改进的OTP检测器模型: %v\n", err)
		fmt.Printf("使用默认参数初始化检测器\n")
		return &ImprovedOTPDetector{
			debug: false,
		}
	}
	return detector
}

// NewDetectorBalancedEnhanced 创建一个平衡增强的OTP检测器
func NewDetectorBalancedEnhanced() *BalancedEnhancedOTPDetector {
	// 获取可能的模型路径
	possiblePaths := []string{
		DefaultBalancedEnhancedModelPath,
		filepath.Join("..", DefaultBalancedEnhancedModelPath),
		filepath.Join("..", "..", DefaultBalancedEnhancedModelPath),
	}

	// 尝试不同的路径
	var detector *BalancedEnhancedOTPDetector
	var err error
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			detector, err = NewBalancedEnhancedOTPDetector(path)
			if err == nil {
				return detector
			}
		}
	}

	// 如果所有路径都失败，使用默认参数
	fmt.Printf("警告: 无法加载平衡增强的OTP检测器模型: %v\n", err)
	fmt.Printf("使用默认参数初始化检测器\n")

	// 创建一个默认的检测器
	detector = &BalancedEnhancedOTPDetector{
		vocabulary:        make(map[string]int),
		binary:            true,
		lowercase:         true,
		modelType:         "nb",
		balanceMethod:     "smote",
		debug:             false,
		decisionThreshold: 0.5,
	}

	// 初始化增强特征工程组件
	detector.initEnhancedFeatures()

	return detector
}

// CreateWordEmbeddingDetector 创建一个基于词嵌入的OTP检测器
// 这是一个工厂函数，与detector_word_embedding.go中的NewWordEmbeddingDetector不同
func CreateWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath string, debug bool) (*WordEmbeddingDetector, error) {
	return NewWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath, debug)
}
