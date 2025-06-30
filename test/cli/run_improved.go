package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	// 获取当前目录
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("无法获取当前目录: %v", err)
	}

	// 检查是否在cli目录
	if filepath.Base(currentDir) != "cli" {
		log.Fatalf("请在cli目录下运行此脚本")
	}

	// 编译改进的检测器
	fmt.Println("编译改进的SVM检测器...")
	cmd := exec.Command("go", "build", "-o", "improved_detector", "improved_main.go")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("编译失败: %v", err)
	}

	// 运行改进的检测器
	fmt.Println("\n运行改进的SVM检测器...")
	cmd = exec.Command("./improved_detector")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("运行失败: %v", err)
	}
}
