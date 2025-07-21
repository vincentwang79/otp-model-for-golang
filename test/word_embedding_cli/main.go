package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/VincentWang/otp_model_for_golang/detector"
)

func main() {
	// 定义命令行参数
	modelPath := flag.String("model", "models/word_embedding_svm_model.joblib", "模型文件路径")
	thresholdPath := flag.String("threshold", "models/word_embedding_svm_threshold.txt", "阈值文件路径")
	embeddingPath := flag.String("embedding", "models/word_embedding_vectors.json", "词嵌入文件路径")
	language := flag.String("lang", "auto", "语言 (auto, en, zh, ru, et)")
	debug := flag.Bool("debug", false, "调试模式")
	interactive := flag.Bool("interactive", false, "交互模式")
	messageFlag := flag.String("message", "", "要检测的短信文本")

	flag.Parse()

	// 创建检测器
	detector, err := detector.NewWordEmbeddingDetector(*modelPath, *thresholdPath, *embeddingPath, *debug)
	if err != nil {
		fmt.Printf("创建检测器失败: %v\n", err)
		os.Exit(1)
	}

	// 设置语言
	detector.SetLanguage(*language)

	// 交互模式
	if *interactive {
		runInteractiveMode(detector)
		return
	}

	// 单条消息模式
	if *messageFlag != "" {
		result := detector.IsOTP(*messageFlag)
		if result {
			fmt.Println("结果: OTP短信")
		} else {
			fmt.Println("结果: 非OTP短信")
		}
		return
	}

	// 如果没有指定交互模式或消息，显示帮助
	fmt.Println("请使用 -interactive 参数进入交互模式，或使用 -message 参数指定要检测的短信")
	flag.Usage()
}

// 运行交互模式
func runInteractiveMode(detector *detector.WordEmbeddingDetector) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("词嵌入OTP检测器 - 交互模式")
	fmt.Println("输入短信内容进行检测，输入'exit'或'quit'退出")

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		text := scanner.Text()
		if text == "exit" || text == "quit" {
			break
		}

		if strings.TrimSpace(text) == "" {
			continue
		}

		result := detector.IsOTP(text)
		if result {
			fmt.Println("结果: OTP短信")
		} else {
			fmt.Println("结果: 非OTP短信")
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "读取输入错误: %v\n", err)
	}
}
