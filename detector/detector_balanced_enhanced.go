package detector

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode"
)

// BalancedEnhancedOTPDetector 结合了数据平衡和增强特征工程的OTP检测器
type BalancedEnhancedOTPDetector struct {
	// 模型参数
	vocabulary        map[string]int
	binary            bool
	lowercase         bool
	otpKeywords       []string
	modelType         string
	balanceMethod     string
	decisionThreshold float64
	debug             bool

	// SVM模型参数
	coef      []float64
	intercept float64

	// 朴素贝叶斯模型参数
	featureLogProb [][]float64
	classLogPrior  []float64

	// 增强特征工程
	otpKeywordsByLang map[string]map[string]float64
	digitPatterns     []*regexp.Regexp
	langDetector      *LanguageDetector
}

// NewBalancedEnhancedOTPDetector 创建一个新的平衡数据增强特征OTP检测器
func NewBalancedEnhancedOTPDetector(paramsPath string) (*BalancedEnhancedOTPDetector, error) {
	// 读取模型参数
	paramsData, err := os.ReadFile(paramsPath)
	if err != nil {
		return nil, fmt.Errorf("读取模型参数文件失败: %v", err)
	}

	// 解析JSON
	var params map[string]interface{}
	if err := json.Unmarshal(paramsData, &params); err != nil {
		return nil, fmt.Errorf("解析模型参数失败: %v", err)
	}

	// 创建检测器
	detector := &BalancedEnhancedOTPDetector{
		vocabulary:    make(map[string]int),
		binary:        getBoolParam(params, "binary", false),
		lowercase:     getBoolParam(params, "lowercase", true),
		modelType:     getStringParam(params, "model_type", "svm"),
		balanceMethod: getStringParam(params, "balance_method", "smote"),
		debug:         false,
	}

	// 设置决策阈值
	if threshold, ok := params["decision_threshold"].(float64); ok {
		detector.decisionThreshold = threshold
	} else {
		// 默认阈值
		detector.decisionThreshold = 0.0
	}

	// 解析词汇表
	if vocabMap, ok := params["vocabulary"].(map[string]interface{}); ok {
		for word, indexFloat := range vocabMap {
			if index, ok := indexFloat.(float64); ok {
				detector.vocabulary[word] = int(index)
			}
		}
	}

	// 解析OTP关键词
	if keywordsArr, ok := params["otp_keywords"].([]interface{}); ok {
		for _, kw := range keywordsArr {
			if keyword, ok := kw.(string); ok {
				detector.otpKeywords = append(detector.otpKeywords, keyword)
			}
		}
	}

	// 根据模型类型解析特定参数
	switch detector.modelType {
	case "svm":
		// 解析SVM系数
		if coefArr, ok := params["coef"].([]interface{}); ok {
			for _, c := range coefArr {
				if coef, ok := c.(float64); ok {
					detector.coef = append(detector.coef, coef)
				}
			}
		}
		// 解析截距
		if intercept, ok := params["intercept"].(float64); ok {
			detector.intercept = intercept
		}

	case "nb":
		// 解析特征对数概率
		if featureLogProbArr, ok := params["feature_log_prob"].([]interface{}); ok {
			detector.featureLogProb = make([][]float64, len(featureLogProbArr))
			for i, row := range featureLogProbArr {
				if rowArr, ok := row.([]interface{}); ok {
					detector.featureLogProb[i] = make([]float64, len(rowArr))
					for j, val := range rowArr {
						if prob, ok := val.(float64); ok {
							detector.featureLogProb[i][j] = prob
						}
					}
				}
			}
		}
		// 解析类别先验概率
		if classLogPriorArr, ok := params["class_log_prior"].([]interface{}); ok {
			for _, p := range classLogPriorArr {
				if prior, ok := p.(float64); ok {
					detector.classLogPrior = append(detector.classLogPrior, prior)
				}
			}
		}

	case "rf":
		// 随机森林模型无法直接导出为JSON参数
		return nil, fmt.Errorf("随机森林模型不支持直接参数导出")
	}

	// 初始化增强特征工程组件
	detector.initEnhancedFeatures()

	return detector, nil
}

// EnableDebug 启用或禁用调试输出
func (d *BalancedEnhancedOTPDetector) EnableDebug(enable bool) {
	d.debug = enable
}

// GetDecisionThreshold 返回决策阈值
func (d *BalancedEnhancedOTPDetector) GetDecisionThreshold() float64 {
	return d.decisionThreshold
}

// initEnhancedFeatures 初始化增强特征工程组件
func (d *BalancedEnhancedOTPDetector) initEnhancedFeatures() {
	// 初始化语言检测器
	d.langDetector = NewLanguageDetector()

	// 初始化多语言OTP关键词
	d.otpKeywordsByLang = make(map[string]map[string]float64)

	// 英文关键词
	d.otpKeywordsByLang["en"] = map[string]float64{
		"code":           2.0,
		"verification":   2.0,
		"verify":         1.5,
		"otp":            3.0,
		"one-time":       2.5,
		"one time":       2.5,
		"password":       1.5,
		"passcode":       2.0,
		"secure":         1.0,
		"security":       1.0,
		"authentication": 2.0,
		"authenticate":   2.0,
		"login":          1.5,
		"signin":         1.5,
		"sign-in":        1.5,
		"confirm":        1.5,
		"validation":     2.0,
		"validate":       1.5,
		"pin":            2.0,
		"token":          1.5,
		"access":         1.0,
		"authorization":  1.5,
		"authorized":     1.0,
		"expires":        1.0,
		"valid for":      1.5,
		"valid until":    1.5,
		"minutes":        0.5,
		"do not share":   1.5,
		"don't share":    1.5,
		"do not reply":   0.5,
	}

	// 中文关键词
	d.otpKeywordsByLang["zh"] = map[string]float64{
		"验证码":    3.0,
		"校验码":    2.5,
		"动态码":    2.5,
		"短信验证码":  3.0,
		"验证":     2.0,
		"一次性密码":  2.5,
		"密码":     1.5,
		"登录":     1.5,
		"登陆":     1.5,
		"安全码":    2.0,
		"有效期":    1.5,
		"分钟":     0.5,
		"请勿泄露":   1.5,
		"请勿告诉他人": 1.5,
		"请勿转发":   1.5,
		"请勿分享":   1.5,
		"请妥善保管":  1.0,
		"请及时输入":  1.0,
		"请在":     0.5,
		"内输入":    0.5,
	}

	// 俄语关键词
	d.otpKeywordsByLang["ru"] = map[string]float64{
		"код":                 3.0,
		"подтверждения":       2.0,
		"проверочный":         2.5,
		"пароль":              1.5,
		"одноразовый":         2.5,
		"авторизации":         1.5,
		"вход":                1.5,
		"действителен":        1.0,
		"минут":               0.5,
		"не сообщайте":        1.5,
		"никому не сообщайте": 1.5,
	}

	// 爱沙尼亚语关键词
	d.otpKeywordsByLang["et"] = map[string]float64{
		"kood":          3.0,
		"kinnituskood":  2.5,
		"salasõna":      1.5,
		"ühekordne":     2.5,
		"parool":        1.5,
		"sisselogimine": 1.5,
		"kehtib":        1.0,
		"minutit":       0.5,
		"ära jaga":      1.5,
	}

	// 初始化数字模式正则表达式
	d.digitPatterns = []*regexp.Regexp{
		regexp.MustCompile(`\b\d{4}\b`),                   // 4位数字
		regexp.MustCompile(`\b\d{5}\b`),                   // 5位数字
		regexp.MustCompile(`\b\d{6}\b`),                   // 6位数字
		regexp.MustCompile(`\b\d{8}\b`),                   // 8位数字
		regexp.MustCompile(`\b\d{3}[- ]\d{3}\b`),          // 带分隔符的6位数字
		regexp.MustCompile(`\b\d{2}[- ]\d{2}[- ]\d{2}\b`), // 带分隔符的6位数字
		regexp.MustCompile(`\(\d{4,6}\)`),                 // 括号中的4-6位数字
		regexp.MustCompile(`\b\d{4,6}[：:]\b`),             // 冒号后的4-6位数字
	}
}

// IsOTP 检测消息是否为OTP短信
func (d *BalancedEnhancedOTPDetector) IsOTP(message string) bool {
	// 检测语言
	lang := d.langDetector.DetectLanguage(message)

	// 对中文消息进行特殊处理
	if lang == "zh" {
		// 检查是否包含OTP关键词
		hasOTPKeyword := false
		for keyword := range d.otpKeywordsByLang["zh"] {
			if strings.Contains(strings.ToLower(message), keyword) {
				hasOTPKeyword = true
				break
			}
		}

		// 检查是否包含数字模式
		hasDigitPattern := false
		for _, pattern := range d.digitPatterns {
			if pattern.MatchString(message) {
				hasDigitPattern = true
				break
			}
		}

		// 如果既没有OTP关键词也没有数字模式，则不是OTP短信
		if !hasOTPKeyword && !hasDigitPattern {
			return false
		}
	}

	// 使用模型进行预测
	score, _ := d.GetOTPScore(message)
	return score >= d.decisionThreshold
}

// GetOTPScore 获取消息的OTP得分
func (d *BalancedEnhancedOTPDetector) GetOTPScore(message string) (float64, map[string]interface{}) {
	// 预处理文本
	processedText := d.preprocessText(message)

	// 提取特征
	features := d.extractFeatures(processedText)

	// 根据模型类型计算得分
	var score float64
	switch d.modelType {
	case "svm":
		score = d.predictSVM(features)
	case "nb":
		score = d.predictNB(features)
	default:
		score = 0.0
	}

	// 调试信息
	debugInfo := make(map[string]interface{})
	if d.debug {
		debugInfo["processed_text"] = processedText
		debugInfo["model_type"] = d.modelType
		debugInfo["balance_method"] = d.balanceMethod
		debugInfo["decision_threshold"] = d.decisionThreshold
		debugInfo["score"] = score
		debugInfo["is_otp"] = score >= d.decisionThreshold
	}

	return score, debugInfo
}

// preprocessText 预处理文本
func (d *BalancedEnhancedOTPDetector) preprocessText(text string) string {
	// 检测语言
	lang := d.langDetector.DetectLanguage(text)

	// 转为小写
	if d.lowercase {
		text = strings.ToLower(text)
	}

	// 提取数字模式
	var digitPatterns []string
	for _, pattern := range d.digitPatterns {
		matches := pattern.FindAllString(text, -1)
		for _, match := range matches {
			digitLen := 0
			for _, r := range match {
				if unicode.IsDigit(r) {
					digitLen++
				}
			}
			digitPatterns = append(digitPatterns, fmt.Sprintf("DIGITS_%d", digitLen))
		}
	}

	// 提取语言特定的关键词
	var keywordPatterns []string
	hasOTPKeyword := false
	if keywords, ok := d.otpKeywordsByLang[lang]; ok {
		for keyword, weight := range keywords {
			if strings.Contains(text, keyword) {
				hasOTPKeyword = true
				// 根据权重重复添加关键词特征
				repeats := int(math.Ceil(weight))
				for i := 0; i < repeats; i++ {
					keywordPatterns = append(keywordPatterns, fmt.Sprintf("KEYWORD_%s_%s", lang, keyword))
				}
			}
		}
	}

	// 对于未识别的语言，尝试使用所有语言的关键词
	if lang == "unknown" {
		for langCode, keywords := range d.otpKeywordsByLang {
			for keyword, weight := range keywords {
				if strings.Contains(text, keyword) {
					hasOTPKeyword = true
					repeats := int(math.Ceil(weight))
					for i := 0; i < repeats; i++ {
						keywordPatterns = append(keywordPatterns, fmt.Sprintf("KEYWORD_%s_%s", langCode, keyword))
					}
				}
			}
		}
	}

	// 添加语言标记
	langPattern := fmt.Sprintf("LANG_%s", lang)

	// 特殊处理中文文本
	if lang == "zh" {
		// 检查是否包含数字模式
		hasDigitPattern := len(digitPatterns) > 0

		// 如果既没有OTP关键词也没有数字模式，则添加多个非OTP标记
		if !hasOTPKeyword && !hasDigitPattern {
			// 添加多个非OTP标记，使其在特征向量中更明显
			for i := 0; i < 10; i++ {
				keywordPatterns = append(keywordPatterns, "NON_OTP_ZH_MARKER")
			}
		}
	}

	// 组合所有特征
	enhancedText := fmt.Sprintf("%s %s %s %s",
		text,
		strings.Join(digitPatterns, " "),
		strings.Join(keywordPatterns, " "),
		langPattern)

	return enhancedText
}

// extractFeatures 从文本中提取特征
func (d *BalancedEnhancedOTPDetector) extractFeatures(text string) map[int]float64 {
	// 分词
	words := strings.Fields(text)

	// 提取n-gram特征
	ngrams := make(map[string]int)

	// 1-gram
	for _, word := range words {
		ngrams[word]++
	}

	// 2-gram
	for i := 0; i < len(words)-1; i++ {
		bigram := words[i] + " " + words[i+1]
		ngrams[bigram]++
	}

	// 3-gram
	for i := 0; i < len(words)-2; i++ {
		trigram := words[i] + " " + words[i+1] + " " + words[i+2]
		ngrams[trigram]++
	}

	// 映射到特征索引
	features := make(map[int]float64)
	for ngram, count := range ngrams {
		if index, ok := d.vocabulary[ngram]; ok {
			if d.binary {
				features[index] = 1.0
			} else {
				features[index] = float64(count)
			}
		}
	}

	return features
}

// predictSVM 使用SVM模型预测
func (d *BalancedEnhancedOTPDetector) predictSVM(features map[int]float64) float64 {
	score := d.intercept

	// 计算特征向量与权重的点积
	for index, value := range features {
		if index < len(d.coef) {
			score += value * d.coef[index]
		}
	}

	return score
}

// predictNB 使用朴素贝叶斯模型预测
func (d *BalancedEnhancedOTPDetector) predictNB(features map[int]float64) float64 {
	if len(d.featureLogProb) != 2 || len(d.classLogPrior) != 2 {
		return 0.0
	}

	// 检查是否为中文非OTP短信
	isNonOTPZH := false
	nonOTPCount := 0
	for _, value := range features {
		// 如果有多个特征值大于1.0，可能是我们添加的NON_OTP_ZH_MARKER
		if value > 1.0 {
			nonOTPCount++
		}
	}

	// 如果有多个这样的特征，很可能是中文非OTP短信
	if nonOTPCount >= 5 {
		isNonOTPZH = true
	}

	// 如果是中文非OTP短信，直接返回低分数
	if isNonOTPZH {
		return 0.1 // 远低于阈值
	}

	// 计算每个类别的对数概率
	logProbs := make([]float64, 2)
	for i := 0; i < 2; i++ {
		logProbs[i] = d.classLogPrior[i]
		for index, value := range features {
			if index < len(d.featureLogProb[i]) {
				logProbs[i] += value * d.featureLogProb[i][index]
			}
		}
	}

	// 计算类别1的概率
	if logProbs[0] > logProbs[1] {
		return math.Exp(logProbs[1] - logProbs[0])
	} else {
		return 1.0 - math.Exp(logProbs[0]-logProbs[1])
	}
}

// 辅助函数 - 获取布尔参数
func getBoolParam(params map[string]interface{}, key string, defaultValue bool) bool {
	if val, ok := params[key].(bool); ok {
		return val
	}
	return defaultValue
}

// 辅助函数 - 获取字符串参数
func getStringParam(params map[string]interface{}, key string, defaultValue string) string {
	if val, ok := params[key].(string); ok {
		return val
	}
	return defaultValue
}
