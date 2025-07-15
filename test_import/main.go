package main

import (
	"fmt"
	"github.com/vincentwang79/otp-model-for-golang/detector"
)

func main() {
	fmt.Println("Testing detector package import")
	
	// 创建一个语言检测器实例（这是detector包中最简单的可以直接实例化的结构）
	langDetector := detector.NewLanguageDetector()
	
	// 使用语言检测器
	text := "Hello, world!"
	lang := langDetector.DetectLanguage(text)
	
	fmt.Printf("Detected language for '%s': %s\n", text, lang)
	
	text2 := "你好，世界！"
	lang2 := langDetector.DetectLanguage(text2)
	
	fmt.Printf("Detected language for '%s': %s\n", text2, lang2)
}
