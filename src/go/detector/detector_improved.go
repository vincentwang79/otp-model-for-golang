package detector

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"strings"
)

// ImprovedModelParams 存储从Python导出的改进模型参数
type ImprovedModelParams struct {
	Vocabulary    map[string]int `json:"vocabulary"`
	Coef          []float64      `json:"coef"`
	Intercept     float64        `json:"intercept"`
	Binary        bool           `json:"binary"`
	Lowercase     bool           `json:"lowercase"`
	OTPKeywords   []string       `json:"otp_keywords"`
	ModelType     string         `json:"model_type"`
	BalanceMethod string         `json:"balance_method"`

	// 朴素贝叶斯模型特有参数
	FeatureLogProb [][]float64 `json:"feature_log_prob"`
	ClassLogPrior  []float64   `json:"class_log_prior"`

	// 决策阈值
	DecisionThreshold float64 `json:"decision_threshold"`
}

// ImprovedOTPDetector 使用平衡后的机器学习模型检测OTP短信
type ImprovedOTPDetector struct {
	params ImprovedModelParams
	debug  bool
}

// NewImprovedOTPDetector 创建新的改进OTP检测器
func NewImprovedOTPDetector(paramsPath string) (*ImprovedOTPDetector, error) {
	// 加载模型参数
	params, err := loadImprovedModelParams(paramsPath)
	if err != nil {
		return nil, fmt.Errorf("加载模型参数失败: %w", err)
	}

	// 验证支持的模型类型
	if params.ModelType != "svm" && params.ModelType != "nb" {
		return nil, fmt.Errorf("不支持的模型类型: %s，目前仅支持SVM和NB模型", params.ModelType)
	}

	// 设置默认决策阈值
	if params.DecisionThreshold == 0 {
		if params.ModelType == "svm" {
			if params.BalanceMethod == "smote" {
				params.DecisionThreshold = 0.1174 // SMOTE平衡的SVM模型最佳阈值
			} else {
				params.DecisionThreshold = 0.2741 // 欠采样平衡的SVM模型最佳阈值
			}
		} else if params.ModelType == "nb" {
			params.DecisionThreshold = 0.8268 // SMOTE平衡的NB模型最佳阈值
		}
	}

	return &ImprovedOTPDetector{
		params: params,
		debug:  false,
	}, nil
}

// EnableDebug 启用/禁用调试模式
func (d *ImprovedOTPDetector) EnableDebug(enable bool) {
	d.debug = enable
}

// IsOTP 判断消息是否为OTP短信
func (d *ImprovedOTPDetector) IsOTP(message string) (bool, float64, error) {
	// 预处理文本
	processedText := d.preprocessText(message)
	if d.params.Lowercase {
		processedText = strings.ToLower(processedText)
	}

	// 提取特征
	features := d.extractFeatures(processedText)

	// 根据模型类型计算分数
	var score float64
	var confidence float64

	if d.params.ModelType == "svm" {
		// 计算SVM分数 (线性SVM决策函数)
		score = d.calculateSVMScore(features)
		confidence = sigmoid(score)
	} else if d.params.ModelType == "nb" {
		// 计算朴素贝叶斯分数
		score, confidence = d.calculateNBScore(features)
	}

	// 使用优化后的决策阈值
	isOTP := confidence > d.params.DecisionThreshold

	if d.debug {
		fmt.Printf("输入文本: %s\n", message)
		fmt.Printf("处理后文本: %s\n", processedText)
		fmt.Printf("模型类型: %s, 平衡方法: %s\n", d.params.ModelType, d.params.BalanceMethod)
		fmt.Printf("分数: %f, 置信度: %f, 决策阈值: %f\n", score, confidence, d.params.DecisionThreshold)
		fmt.Printf("判断: %v\n", isOTP)
	}

	return isOTP, confidence, nil
}

// preprocessText 预处理文本，提取数字模式和关键词
func (d *ImprovedOTPDetector) preprocessText(text string) string {
	// 清洗文本
	cleanedText := cleanText(text)

	// 提取数字模式
	digits := extractDigits(text)
	digitPattern := ""
	for _, digit := range digits {
		digitPattern += fmt.Sprintf(" DIGITS_%d", len(digit))
	}

	// 提取关键词
	keywordPattern := ""
	for _, keyword := range d.params.OTPKeywords {
		if strings.Contains(strings.ToLower(text), strings.ToLower(keyword)) {
			keywordPattern += fmt.Sprintf(" KEYWORD_%s", strings.ToLower(keyword))
		}
	}

	// 结合原始文本和提取的模式
	return cleanedText + digitPattern + keywordPattern
}

// extractFeatures 从文本中提取TF-IDF特征
func (d *ImprovedOTPDetector) extractFeatures(text string) map[int]float64 {
	// 分词
	words := strings.Fields(text)

	// 计算词频
	wordCounts := make(map[string]int)
	for _, word := range words {
		wordCounts[word]++
	}

	// 提取2-gram和3-gram特征
	for i := 0; i < len(words)-1; i++ {
		// 2-gram
		bigram := words[i] + " " + words[i+1]
		wordCounts[bigram]++

		// 3-gram
		if i < len(words)-2 {
			trigram := words[i] + " " + words[i+1] + " " + words[i+2]
			wordCounts[trigram]++
		}
	}

	// 创建TF-IDF特征向量
	features := make(map[int]float64)
	for word, count := range wordCounts {
		if idx, exists := d.params.Vocabulary[word]; exists {
			if d.params.Binary {
				features[idx] = 1.0
			} else {
				// 使用次线性TF缩放 (1 + log(tf))
				features[idx] = 1.0 + math.Log(float64(count))
			}
		}
	}

	return features
}

// calculateSVMScore 计算线性SVM分数
func (d *ImprovedOTPDetector) calculateSVMScore(features map[int]float64) float64 {
	score := d.params.Intercept

	for idx, value := range features {
		if idx < len(d.params.Coef) {
			score += value * d.params.Coef[idx]
		}
	}

	return score
}

// calculateNBScore 计算朴素贝叶斯分数和置信度
func (d *ImprovedOTPDetector) calculateNBScore(features map[int]float64) (float64, float64) {
	// 计算每个类别的对数概率
	logProbs := make([]float64, 2)

	// 添加先验概率
	for i := 0; i < 2; i++ {
		logProbs[i] = d.params.ClassLogPrior[i]
	}

	// 添加特征对数概率
	for idx, value := range features {
		if idx < len(d.params.FeatureLogProb[0]) {
			// 对于每个存在的特征，累加其对数概率
			for i := 0; i < 2; i++ {
				if d.params.Binary {
					// 二进制特征
					logProbs[i] += d.params.FeatureLogProb[i][idx]
				} else {
					// TF-IDF特征
					logProbs[i] += value * d.params.FeatureLogProb[i][idx]
				}
			}
		}
	}

	// 计算softmax概率
	maxLogProb := math.Max(logProbs[0], logProbs[1])
	expProbs := make([]float64, 2)
	var sumExpProb float64

	for i := 0; i < 2; i++ {
		expProbs[i] = math.Exp(logProbs[i] - maxLogProb)
		sumExpProb += expProbs[i]
	}

	// 归一化概率
	probs := make([]float64, 2)
	for i := 0; i < 2; i++ {
		probs[i] = expProbs[i] / sumExpProb
	}

	// 返回分数（两个类别对数概率之差）和OTP类别的概率
	score := logProbs[1] - logProbs[0]
	confidence := probs[1]

	return score, confidence
}

// loadImprovedModelParams 从JSON文件加载改进模型参数
func loadImprovedModelParams(path string) (ImprovedModelParams, error) {
	var params ImprovedModelParams

	// 读取文件
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return params, fmt.Errorf("读取模型参数文件失败: %w", err)
	}

	// 解析JSON
	err = json.Unmarshal(data, &params)
	if err != nil {
		return params, fmt.Errorf("解析模型参数JSON失败: %w", err)
	}

	return params, nil
}
