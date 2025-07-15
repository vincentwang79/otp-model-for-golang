package detector

import (
	"regexp"
	"strings"
	"unicode"
)

// LanguageDetector 语言检测器
type LanguageDetector struct {
	// 语言特征正则表达式
	chinesePattern  *regexp.Regexp
	russianPattern  *regexp.Regexp
	estonianPattern *regexp.Regexp
}

// NewLanguageDetector 创建一个新的语言检测器
func NewLanguageDetector() *LanguageDetector {
	return &LanguageDetector{
		// 中文字符范围
		chinesePattern: regexp.MustCompile(`[\p{Han}]`),
		// 俄语字符 (西里尔字母)
		russianPattern: regexp.MustCompile(`[А-Яа-я]`),
		// 爱沙尼亚语特有字符
		estonianPattern: regexp.MustCompile(`[õäöüÕÄÖÜ]`),
	}
}

// DetectLanguage 检测文本语言
// 返回语言代码: "en" (英语), "zh" (中文), "ru" (俄语), "et" (爱沙尼亚语), "unknown" (未知)
func (d *LanguageDetector) DetectLanguage(text string) string {
	// 计算各种语言字符的数量
	chineseCount := len(d.chinesePattern.FindAllString(text, -1))
	russianCount := len(d.russianPattern.FindAllString(text, -1))
	estonianCount := len(d.estonianPattern.FindAllString(text, -1))

	// 计算拉丁字符数量
	latinCount := 0
	totalLetters := 0

	for _, r := range text {
		if unicode.IsLetter(r) {
			totalLetters++
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				latinCount++
			}
		}
	}

	// 如果文本太短，可能不可靠
	if totalLetters < 3 {
		return "unknown"
	}

	// 根据字符数量比例判断语言
	if chineseCount > 0 && float64(chineseCount)/float64(totalLetters) > 0.15 {
		return "zh"
	}

	if russianCount > 0 && float64(russianCount)/float64(totalLetters) > 0.15 {
		return "ru"
	}

	if estonianCount > 0 && float64(estonianCount)/float64(totalLetters) > 0.05 {
		return "et"
	}

	if latinCount > 0 && float64(latinCount)/float64(totalLetters) > 0.5 {
		return "en"
	}

	// 检查英语关键词
	englishKeywords := []string{" the ", " and ", " for ", " your ", " you ", " with ", " this ", " that ", " have ", " from "}
	textLower := " " + strings.ToLower(text) + " "

	for _, keyword := range englishKeywords {
		if strings.Contains(textLower, keyword) {
			return "en"
		}
	}

	// 如果没有明显特征，根据最多的字符类型判断
	counts := map[string]int{
		"zh": chineseCount,
		"ru": russianCount,
		"et": estonianCount,
		"en": latinCount,
	}

	maxLang := "unknown"
	maxCount := 0

	for lang, count := range counts {
		if count > maxCount {
			maxLang = lang
			maxCount = count
		}
	}

	return maxLang
}

// DetectLanguageWithConfidence 检测文本语言并返回置信度
func (d *LanguageDetector) DetectLanguageWithConfidence(text string) (string, float64) {
	// 计数不同语言的特征
	chineseCount := len(d.chinesePattern.FindAllString(text, -1))
	russianCount := len(d.russianPattern.FindAllString(text, -1))
	estonianCount := len(d.estonianPattern.FindAllString(text, -1))

	// 计算拉丁字符数量
	latinCount := 0
	totalLetters := 0

	for _, r := range text {
		if unicode.IsLetter(r) {
			totalLetters++
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				latinCount++
			}
		}
	}

	// 如果文本太短，可能不可靠
	if totalLetters < 5 {
		return "unknown", 0.3
	}

	// 计算各语言的置信度
	confidences := map[string]float64{
		"zh": float64(chineseCount) / float64(totalLetters),
		"ru": float64(russianCount) / float64(totalLetters),
		"et": float64(estonianCount) / float64(totalLetters),
		"en": float64(latinCount) / float64(totalLetters),
	}

	// 找出最高置信度的语言
	maxLang := "unknown"
	maxConf := 0.3 // 最低置信度阈值

	for lang, conf := range confidences {
		if conf > maxConf {
			maxLang = lang
			maxConf = conf
		}
	}

	return maxLang, maxConf
}
