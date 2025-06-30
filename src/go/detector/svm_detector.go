package detector

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"regexp"
	"strings"
)

// SVMModelParams 存储从Python导出的SVM模型参数
type SVMModelParams struct {
	Vocabulary  map[string]int `json:"vocabulary"`
	Coef        []float64      `json:"coef"`
	Intercept   float64        `json:"intercept"`
	Binary      bool           `json:"binary"`
	Lowercase   bool           `json:"lowercase"`
	OTPKeywords []string       `json:"otp_keywords"`
	ModelType   string         `json:"model_type"`
}

// SVMDetector 使用SVM模型检测OTP短信
type SVMDetector struct {
	params SVMModelParams
	debug  bool
}

// NewSVMDetector 创建新的SVM检测器
func NewSVMDetector(paramsPath string) (*SVMDetector, error) {
	// 加载模型参数
	params, err := loadSVMModelParams(paramsPath)
	if err != nil {
		return nil, fmt.Errorf("加载模型参数失败: %w", err)
	}

	// 验证是否为SVM模型
	if params.ModelType != "svm" {
		return nil, fmt.Errorf("不支持的模型类型: %s，目前仅支持SVM模型", params.ModelType)
	}

	return &SVMDetector{
		params: params,
		debug:  false,
	}, nil
}

// EnableDebug 启用/禁用调试模式
func (d *SVMDetector) EnableDebug(enable bool) {
	d.debug = enable
}

// IsOTP 判断消息是否为OTP短信
func (d *SVMDetector) IsOTP(message string) (bool, float64, error) {
	// 预处理文本
	processedText := d.preprocessText(message)
	if d.params.Lowercase {
		processedText = strings.ToLower(processedText)
	}

	// 提取特征
	features := d.extractFeatures(processedText)

	// 计算分数 (线性SVM决策函数)
	score := d.calculateScore(features)

	// 分数大于0表示OTP类
	isOTP := score > 0
	confidence := svmSigmoid(score)

	if d.debug {
		fmt.Printf("输入文本: %s\n", message)
		fmt.Printf("处理后文本: %s\n", processedText)
		fmt.Printf("分数: %f, 置信度: %f\n", score, confidence)
		fmt.Printf("判断: %v\n", isOTP)
	}

	return isOTP, confidence, nil
}

// preprocessText 预处理文本，提取数字模式和关键词
func (d *SVMDetector) preprocessText(text string) string {
	// 清洗文本
	cleanedText := svmCleanText(text)

	// 提取数字模式
	digits := svmExtractDigits(text)
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
func (d *SVMDetector) extractFeatures(text string) map[int]float64 {
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

// calculateScore 计算线性SVM分数
func (d *SVMDetector) calculateScore(features map[int]float64) float64 {
	score := d.params.Intercept

	for idx, value := range features {
		if idx < len(d.params.Coef) {
			score += value * d.params.Coef[idx]
		}
	}

	return score
}

// loadSVMModelParams 从JSON文件加载模型参数
func loadSVMModelParams(path string) (SVMModelParams, error) {
	var params SVMModelParams

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

// svmCleanText 清洗文本
func svmCleanText(text string) string {
	// 替换标点符号为空格
	re := regexp.MustCompile(`[^\w\s]`)
	text = re.ReplaceAllString(text, " ")

	// 标准化空格
	reSpace := regexp.MustCompile(`\s+`)
	text = reSpace.ReplaceAllString(text, " ")

	// 去除首尾空格
	text = strings.TrimSpace(text)

	return text
}

// svmExtractDigits 提取文本中的数字
func svmExtractDigits(text string) []string {
	re := regexp.MustCompile(`\d+`)
	return re.FindAllString(text, -1)
}

// svmSigmoid 函数，将分数转换为0-1之间的置信度
func svmSigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
