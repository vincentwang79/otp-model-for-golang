package detector

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"regexp"
	"strings"
	"unicode"
)

// EnhancedModelParams 存储从Python导出的增强模型参数
type EnhancedModelParams struct {
	Vocabulary        map[string]int `json:"vocabulary"`
	Coef              []float64      `json:"coef"`
	Intercept         float64        `json:"intercept"`
	Binary            bool           `json:"binary"`
	Lowercase         bool           `json:"lowercase"`
	OTPKeywords       []string       `json:"otp_keywords"`
	ModelType         string         `json:"model_type"`
	BalanceMethod     string         `json:"balance_method"`
	DecisionThreshold float64        `json:"decision_threshold"`

	// 朴素贝叶斯模型特有参数
	FeatureLogProb [][]float64 `json:"feature_log_prob"`
	ClassLogPrior  []float64   `json:"class_log_prior"`
}

// EnhancedOTPDetector 使用增强特征工程的OTP检测器
type EnhancedOTPDetector struct {
	params EnhancedModelParams
	debug  bool

	// 数字模式正则表达式
	digitPatterns map[string]*regexp.Regexp

	// 关键词映射
	keywordsByLang map[string]map[string]float64
}

// NewEnhancedOTPDetector 创建新的增强OTP检测器
func NewEnhancedOTPDetector(paramsPath string) (*EnhancedOTPDetector, error) {
	// 加载模型参数
	params, err := loadEnhancedModelParams(paramsPath)
	if err != nil {
		return nil, fmt.Errorf("加载模型参数失败: %w", err)
	}

	// 验证支持的模型类型
	if params.ModelType != "svm" && params.ModelType != "nb" && params.ModelType != "rf" {
		return nil, fmt.Errorf("不支持的模型类型: %s，目前仅支持SVM、NB和RF模型", params.ModelType)
	}

	// 设置默认决策阈值
	if params.DecisionThreshold == 0 {
		params.DecisionThreshold = 0.5
	}

	// 初始化数字模式正则表达式
	digitPatterns := make(map[string]*regexp.Regexp)
	digitPatterns["has_4_digits"] = regexp.MustCompile(`\b\d{4}\b`)
	digitPatterns["has_5_digits"] = regexp.MustCompile(`\b\d{5}\b`)
	digitPatterns["has_6_digits"] = regexp.MustCompile(`\b\d{6}\b`)
	digitPatterns["has_8_digits"] = regexp.MustCompile(`\b\d{8}\b`)
	digitPatterns["has_consecutive_digits"] = regexp.MustCompile(`\d{2,}`)
	digitPatterns["has_digits_with_separator"] = regexp.MustCompile(`\d+[\s\-\.]+\d+`)
	digitPatterns["has_digits_in_brackets"] = regexp.MustCompile(`[\[\(（【]\s*\d+\s*[\]\)）】]`)
	digitPatterns["has_digits_after_colon"] = regexp.MustCompile(`[:：]\s*\d+`)
	digitPatterns["has_digits_with_prefix"] = regexp.MustCompile(`[a-zA-Z]+\d+`)

	// 初始化关键词映射
	keywordsByLang := initKeywordsByLang()

	return &EnhancedOTPDetector{
		params:         params,
		debug:          false,
		digitPatterns:  digitPatterns,
		keywordsByLang: keywordsByLang,
	}, nil
}

// EnableDebug 启用/禁用调试模式
func (d *EnhancedOTPDetector) EnableDebug(enable bool) {
	d.debug = enable
}

// IsOTP 判断消息是否为OTP短信
func (d *EnhancedOTPDetector) IsOTP(message string) (bool, float64, error) {
	// 预处理文本，提取增强特征
	processedText, language := d.preprocessText(message)
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
	} else if d.params.ModelType == "rf" {
		// 随机森林不容易在Go中实现，这里使用SVM方式计算
		score = d.calculateSVMScore(features)
		confidence = sigmoid(score)
	}

	// 使用优化后的决策阈值
	isOTP := confidence > d.params.DecisionThreshold

	if d.debug {
		fmt.Printf("输入文本: %s\n", message)
		fmt.Printf("处理后文本: %s\n", processedText)
		fmt.Printf("检测到语言: %s\n", language)
		fmt.Printf("模型类型: %s, 特征工程: %s\n", d.params.ModelType, d.params.BalanceMethod)
		fmt.Printf("分数: %f, 置信度: %f, 决策阈值: %f\n", score, confidence, d.params.DecisionThreshold)
		fmt.Printf("判断: %v\n", isOTP)
	}

	return isOTP, confidence, nil
}

// preprocessText 预处理文本，提取增强特征
func (d *EnhancedOTPDetector) preprocessText(text string) (string, string) {
	// 检测语言
	language := d.detectLanguage(text)

	// 基本清洗
	cleanedText := d.cleanText(text)

	// 提取数字模式特征
	digitFeatures := d.extractDigitPatterns(text)

	// 提取关键词特征
	keywordFeatures := d.extractKeywordFeatures(text, language)

	// 构建特征字符串
	featureStr := cleanedText

	// 添加数字模式特征
	for pattern, matches := range digitFeatures {
		if matches {
			featureStr += fmt.Sprintf(" DIGIT_PATTERN_%s", pattern)
		}
	}

	// 添加数字百分比特征
	digitPercentage := d.calculateDigitPercentage(text)
	if digitPercentage > 0.3 {
		featureStr += " DIGIT_PATTERN_high_percentage"
	} else if digitPercentage > 0.1 {
		featureStr += " DIGIT_PATTERN_medium_percentage"
	} else if digitPercentage > 0 {
		featureStr += " DIGIT_PATTERN_low_percentage"
	}

	// 添加关键词特征
	for keyword, weight := range keywordFeatures {
		// 根据权重添加多次关键词特征，增强其影响
		repeat := int(weight * 2) // 权重转换为重复次数
		for i := 0; i < repeat; i++ {
			featureStr += fmt.Sprintf(" KEYWORD_%s", keyword)
		}
	}

	// 添加语言标识特征
	featureStr += fmt.Sprintf(" LANG_%s", language)

	return featureStr, language
}

// detectLanguage 检测文本语言
func (d *EnhancedOTPDetector) detectLanguage(text string) string {
	// 简单的语言检测实现
	// 检查中文字符
	chineseChars := 0
	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			chineseChars++
		}
	}
	if float64(chineseChars)/float64(len(text)) > 0.3 {
		return "zh"
	}

	// 检查俄语字符 (西里尔字母)
	russianChars := 0
	for _, r := range text {
		if unicode.Is(unicode.Cyrillic, r) {
			russianChars++
		}
	}
	if float64(russianChars)/float64(len(text)) > 0.3 {
		return "ru"
	}

	// 检查爱沙尼亚语特殊字符
	estonianChars := 0
	estonianSpecialChars := map[rune]bool{
		'õ': true, 'ä': true, 'ö': true, 'ü': true,
		'Õ': true, 'Ä': true, 'Ö': true, 'Ü': true,
	}
	for _, r := range text {
		if estonianSpecialChars[r] {
			estonianChars++
		}
	}
	if float64(estonianChars)/float64(len(text)) > 0.05 {
		return "et"
	}

	// 默认为英文
	return "en"
}

// cleanText 清洗文本
func (d *EnhancedOTPDetector) cleanText(text string) string {
	// 转为小写
	text = strings.ToLower(text)

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

// extractDigitPatterns 提取数字模式特征
func (d *EnhancedOTPDetector) extractDigitPatterns(text string) map[string]bool {
	features := make(map[string]bool)

	// 提取各种数字模式
	for patternName, pattern := range d.digitPatterns {
		matches := pattern.FindAllString(text, -1)
		features[patternName] = len(matches) > 0
	}

	return features
}

// calculateDigitPercentage 计算数字在文本中的比例
func (d *EnhancedOTPDetector) calculateDigitPercentage(text string) float64 {
	if len(text) == 0 {
		return 0
	}

	digitCount := 0
	for _, r := range text {
		if unicode.IsDigit(r) {
			digitCount++
		}
	}

	return float64(digitCount) / float64(len(text))
}

// extractKeywordFeatures 提取关键词特征
func (d *EnhancedOTPDetector) extractKeywordFeatures(text string, language string) map[string]float64 {
	textLower := strings.ToLower(text)
	features := make(map[string]float64)

	// 获取对应语言的关键词列表
	keywords, ok := d.keywordsByLang[language]
	if !ok {
		keywords = d.keywordsByLang["en"] // 默认使用英文关键词
	}

	// 检查每个关键词
	for keyword, weight := range keywords {
		if strings.Contains(textLower, keyword) {
			features[keyword] = weight
		}
	}

	// 对于非英文语言，也检查英文关键词
	if language != "en" {
		for keyword, weight := range d.keywordsByLang["en"] {
			if strings.Contains(textLower, keyword) {
				features[keyword] = weight
			}
		}
	}

	return features
}

// extractFeatures 从文本中提取特征
func (d *EnhancedOTPDetector) extractFeatures(text string) map[int]float64 {
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

	// 创建特征向量
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
func (d *EnhancedOTPDetector) calculateSVMScore(features map[int]float64) float64 {
	score := d.params.Intercept

	for idx, value := range features {
		if idx < len(d.params.Coef) {
			score += value * d.params.Coef[idx]
		}
	}

	return score
}

// calculateNBScore 计算朴素贝叶斯分数和置信度
func (d *EnhancedOTPDetector) calculateNBScore(features map[int]float64) (float64, float64) {
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

// initKeywordsByLang 初始化各语言的关键词映射
func initKeywordsByLang() map[string]map[string]float64 {
	keywordsByLang := make(map[string]map[string]float64)

	// 英文关键词及其权重
	keywordsByLang["en"] = map[string]float64{
		"verification":  2.0,
		"code":          1.8,
		"otp":           2.5,
		"one-time":      2.2,
		"password":      1.5,
		"authenticate":  2.0,
		"secure":        1.3,
		"confirm":       1.2,
		"login":         1.5,
		"access":        1.2,
		"valid":         1.3,
		"expires":       1.4,
		"minutes":       1.1,
		"security":      1.4,
		"pin":           1.8,
		"token":         1.7,
		"authorization": 1.5,
		"verify":        2.0,
	}

	// 中文关键词及其权重
	keywordsByLang["zh"] = map[string]float64{
		"验证码": 2.5,
		"验证":  1.8,
		"密码":  1.5,
		"一次性": 2.0,
		"有效期": 1.4,
		"失效":  1.3,
		"分钟":  1.1,
		"登录":  1.5,
		"登陆":  1.5,
		"安全":  1.2,
		"校验":  1.7,
		"确认":  1.3,
		"短信":  1.1,
		"动态码": 2.0,
		"授权码": 1.8,
		"识别码": 1.7,
		"认证码": 2.0,
		"临时":  1.4,
		"勿泄露": 1.6,
		"请勿":  1.2,
	}

	// 俄语关键词及其权重
	keywordsByLang["ru"] = map[string]float64{
		"код":            2.5, // 验证码
		"подтверждения":  2.0, // 确认
		"пароль":         1.8, // 密码
		"одноразовый":    2.2, // 一次性
		"проверки":       1.8, // 验证
		"авторизации":    1.7, // 授权
		"действителен":   1.4, // 有效
		"минут":          1.1, // 分钟
		"безопасности":   1.3, // 安全
		"вход":           1.5, // 登录
		"доступ":         1.4, // 访问
		"секретный":      1.6, // 秘密
		"временный":      1.5, // 临时
		"аутентификации": 1.9, // 认证
		"срок":           1.3, // 期限
		"истекает":       1.4, // 过期
	}

	// 爱沙尼亚语关键词及其权重
	keywordsByLang["et"] = map[string]float64{
		"kood":          2.5, // 验证码
		"kinnituskood":  2.2, // 确认码
		"parool":        1.8, // 密码
		"ühekordne":     2.0, // 一次性
		"turva":         1.5, // 安全
		"kehtib":        1.4, // 有效
		"minutit":       1.1, // 分钟
		"sisselogimine": 1.7, // 登录
		"juurdepääs":    1.5, // 访问
		"kinnitus":      1.8, // 确认
		"autentimine":   2.0, // 认证
		"ajutine":       1.4, // 临时
		"aegub":         1.3, // 过期
		"turvakood":     1.9, // 安全码
		"salajane":      1.6, // 秘密
		"tõendamine":    1.7, // 验证
	}

	return keywordsByLang
}

// loadEnhancedModelParams 从JSON文件加载增强模型参数
func loadEnhancedModelParams(path string) (EnhancedModelParams, error) {
	var params EnhancedModelParams

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
