package detector

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// WordEmbeddingDetector 实现了基于词嵌入的OTP检测器
type WordEmbeddingDetector struct {
	// 模型参数
	modelPath       string
	thresholdPath   string
	embeddingPath   string
	threshold       float64
	debug           bool
	language        string
	embeddings      map[string][]float64
	embeddingDim    int
	otpKeywords     map[string][]string
	otpPatternRegex *regexp.Regexp
}

// NewWordEmbeddingDetector 创建一个新的词嵌入检测器
func NewWordEmbeddingDetector(modelPath, thresholdPath, embeddingPath string, debug bool) (*WordEmbeddingDetector, error) {
	detector := &WordEmbeddingDetector{
		modelPath:     modelPath,
		thresholdPath: thresholdPath,
		embeddingPath: embeddingPath,
		debug:         debug,
		language:      "auto",
		embeddingDim:  300, // 默认嵌入维度
	}

	// 加载阈值
	err := detector.loadThreshold()
	if err != nil {
		return nil, fmt.Errorf("加载阈值失败: %v", err)
	}

	// 加载词嵌入
	err = detector.loadEmbeddings()
	if err != nil {
		return nil, fmt.Errorf("加载词嵌入失败: %v", err)
	}

	// 初始化OTP关键词
	detector.initOTPKeywords()

	// 初始化OTP模式正则表达式
	detector.otpPatternRegex = regexp.MustCompile(`\b\d{4,8}\b`)

	return detector, nil
}

// loadThreshold 加载决策阈值
func (d *WordEmbeddingDetector) loadThreshold() error {
	// 检查文件是否存在
	if _, err := os.Stat(d.thresholdPath); os.IsNotExist(err) {
		// 如果文件不存在，使用默认阈值
		d.threshold = 0.5
		if d.debug {
			fmt.Printf("阈值文件不存在，使用默认阈值: %.2f\n", d.threshold)
		}
		return nil
	}

	// 读取阈值文件
	thresholdBytes, err := ioutil.ReadFile(d.thresholdPath)
	if err != nil {
		return err
	}

	// 解析阈值
	thresholdStr := strings.TrimSpace(string(thresholdBytes))
	d.threshold, err = parseFloat(thresholdStr)
	if err != nil {
		return fmt.Errorf("解析阈值失败: %v", err)
	}

	if d.debug {
		fmt.Printf("加载阈值: %.2f\n", d.threshold)
	}

	return nil
}

// loadEmbeddings 加载词嵌入
func (d *WordEmbeddingDetector) loadEmbeddings() error {
	// 检查文件是否存在
	if _, err := os.Stat(d.embeddingPath); os.IsNotExist(err) {
		// 如果文件不存在，使用空的嵌入
		d.embeddings = make(map[string][]float64)
		if d.debug {
			fmt.Println("词嵌入文件不存在，使用空的嵌入")
		}
		return nil
	}

	// 读取词嵌入文件
	embeddingBytes, err := ioutil.ReadFile(d.embeddingPath)
	if err != nil {
		return err
	}

	// 解析词嵌入
	err = json.Unmarshal(embeddingBytes, &d.embeddings)
	if err != nil {
		return fmt.Errorf("解析词嵌入失败: %v", err)
	}

	// 获取嵌入维度
	for _, vec := range d.embeddings {
		d.embeddingDim = len(vec)
		break
	}

	if d.debug {
		fmt.Printf("加载词嵌入: %d个词, 维度: %d\n", len(d.embeddings), d.embeddingDim)
	}

	return nil
}

// initOTPKeywords 初始化OTP关键词
func (d *WordEmbeddingDetector) initOTPKeywords() {
	d.otpKeywords = map[string][]string{
		"en": {"code", "verification", "verify", "otp", "password", "pin", "authenticate", "login"},
		"zh": {"验证码", "校验码", "确认码", "动态密码", "验证", "登录", "登陆", "密码"},
		"ru": {"код", "подтверждение", "пароль", "пин", "авторизация", "вход"},
		"et": {"kood", "kinnituskood", "parool", "sisselogimine"},
	}
}

// detectLanguage 检测文本语言
func (d *WordEmbeddingDetector) detectLanguage(text string) string {
	// 检测中文
	for _, r := range text {
		if r >= '\u4e00' && r <= '\u9fff' {
			return "zh"
		}
	}

	// 检测俄语 (西里尔字母)
	for _, r := range text {
		if r >= '\u0400' && r <= '\u04FF' {
			return "ru"
		}
	}

	// 检测爱沙尼亚语 (特殊字符)
	estonianChars := "õäöüÕÄÖÜ"
	for _, r := range text {
		if strings.ContainsRune(estonianChars, r) {
			return "et"
		}
	}

	// 默认为英语
	return "en"
}

// preprocessText 预处理文本
func (d *WordEmbeddingDetector) preprocessText(text, language string) []string {
	// 转换为小写
	text = strings.ToLower(text)

	// 移除标点符号
	text = regexp.MustCompile(`[^\w\s]`).ReplaceAllString(text, " ")

	// 分词
	var tokens []string
	if language == "zh" {
		// 中文分词
		// 在Go中实现中文分词较为复杂，这里简化处理
		// 实际应用中可以考虑使用外部分词工具或服务
		tokens = []string{}
		for _, r := range text {
			if r >= '\u4e00' && r <= '\u9fff' {
				tokens = append(tokens, string(r))
			}
		}
	} else {
		// 其他语言按空格分词
		tokens = strings.Fields(text)
	}

	return tokens
}

// extractNumberPatterns 提取数字模式特征
func (d *WordEmbeddingDetector) extractNumberPatterns(text string) map[string]float64 {
	features := make(map[string]float64)

	// 提取4-8位数字
	otpPatterns := d.otpPatternRegex.FindAllString(text, -1)
	if len(otpPatterns) > 0 {
		features["has_potential_otp"] = 1.0
	} else {
		features["has_potential_otp"] = 0.0
	}
	features["otp_pattern_count"] = float64(len(otpPatterns))

	// 数字密度
	digitCount := 0
	for _, r := range text {
		if r >= '0' && r <= '9' {
			digitCount++
		}
	}
	textLen := len(text)
	if textLen > 0 {
		features["digit_density"] = float64(digitCount) / float64(textLen)
	} else {
		features["digit_density"] = 0.0
	}

	// 数字模式的位置
	if len(otpPatterns) > 0 {
		// 找出第一个OTP模式在文本中的相对位置
		firstPos := strings.Index(text, otpPatterns[0])
		if textLen > 0 {
			features["otp_relative_position"] = float64(firstPos) / float64(textLen)
		} else {
			features["otp_relative_position"] = -1.0
		}
	} else {
		features["otp_relative_position"] = -1.0
	}

	return features
}

// extractKeywordFeatures 提取关键词特征
func (d *WordEmbeddingDetector) extractKeywordFeatures(tokens []string, language string) map[string]float64 {
	features := make(map[string]float64)

	// 获取对应语言的关键词列表
	keywords, ok := d.otpKeywords[language]
	if !ok {
		keywords = d.otpKeywords["en"]
	}

	// 计算关键词出现次数
	keywordCount := 0
	for _, token := range tokens {
		for _, keyword := range keywords {
			if token == keyword {
				keywordCount++
				break
			}
		}
	}

	features["otp_keyword_count"] = float64(keywordCount)
	if keywordCount > 0 {
		features["has_otp_keyword"] = 1.0
	} else {
		features["has_otp_keyword"] = 0.0
	}

	// 关键词密度
	if len(tokens) > 0 {
		features["keyword_density"] = float64(keywordCount) / float64(len(tokens))
	} else {
		features["keyword_density"] = 0.0
	}

	return features
}

// getWordEmbedding 获取词的嵌入向量
func (d *WordEmbeddingDetector) getWordEmbedding(word string) []float64 {
	// 查找词嵌入
	if vec, ok := d.embeddings[word]; ok {
		return vec
	}

	// 如果找不到，返回零向量
	return make([]float64, d.embeddingDim)
}

// getTextEmbedding 获取文本的嵌入向量
func (d *WordEmbeddingDetector) getTextEmbedding(tokens []string) []float64 {
	if len(tokens) == 0 {
		return make([]float64, d.embeddingDim)
	}

	// 获取每个词的嵌入向量
	var wordEmbeddings [][]float64
	for _, token := range tokens {
		wordEmbeddings = append(wordEmbeddings, d.getWordEmbedding(token))
	}

	// 计算平均词向量作为文本表示
	textEmbedding := make([]float64, d.embeddingDim)
	for _, wordVec := range wordEmbeddings {
		for i, val := range wordVec {
			textEmbedding[i] += val
		}
	}

	// 归一化
	for i := range textEmbedding {
		textEmbedding[i] /= float64(len(tokens))
	}

	return textEmbedding
}

// extractFeatures 提取所有特征
func (d *WordEmbeddingDetector) extractFeatures(text string) map[string]float64 {
	// 检测语言
	language := d.language
	if language == "auto" {
		language = d.detectLanguage(text)
	}

	// 预处理文本
	tokens := d.preprocessText(text, language)

	// 提取数字模式特征
	numberFeatures := d.extractNumberPatterns(text)

	// 提取关键词特征
	keywordFeatures := d.extractKeywordFeatures(tokens, language)

	// 提取词嵌入特征
	embedding := d.getTextEmbedding(tokens)

	// 合并所有特征
	features := make(map[string]float64)
	for k, v := range numberFeatures {
		features[k] = v
	}
	for k, v := range keywordFeatures {
		features[k] = v
	}

	// 将词嵌入特征添加到特征字典
	for i, value := range embedding {
		features[fmt.Sprintf("embedding_%d", i)] = value
	}

	return features
}

// predictProbability 预测OTP概率
func (d *WordEmbeddingDetector) predictProbability(features map[string]float64) float64 {
	// 注意：这里是一个简化实现
	// 实际应用中应该加载训练好的模型进行预测
	// 这里使用一个简单的规则来模拟预测

	// 计算OTP可能性得分
	score := 0.0

	// 数字模式特征权重
	if features["has_potential_otp"] > 0 {
		score += 0.4
	}
	score += features["otp_pattern_count"] * 0.1
	score += features["digit_density"] * 0.2

	// 关键词特征权重
	if features["has_otp_keyword"] > 0 {
		score += 0.3
	}
	score += features["otp_keyword_count"] * 0.1
	score += features["keyword_density"] * 0.2

	// 归一化到0-1之间
	probability := math.Min(math.Max(score, 0.0), 1.0)

	return probability
}

// IsOTP 判断文本是否为OTP短信
func (d *WordEmbeddingDetector) IsOTP(text string) bool {
	// 提取特征
	features := d.extractFeatures(text)

	// 预测概率
	probability := d.predictProbability(features)

	if d.debug {
		fmt.Printf("文本: %s\n", text)
		fmt.Printf("概率: %.4f, 阈值: %.2f\n", probability, d.threshold)
	}

	// 根据阈值判断
	return probability >= d.threshold
}

// SetDebug 设置调试模式
func (d *WordEmbeddingDetector) SetDebug(debug bool) {
	d.debug = debug
}

// SetLanguage 设置语言
func (d *WordEmbeddingDetector) SetLanguage(language string) {
	d.language = language
}

// 辅助函数：解析浮点数
func parseFloat(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}
