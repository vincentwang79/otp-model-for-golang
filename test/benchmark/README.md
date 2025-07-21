# 词嵌入检测器基准测试

本目录包含用于测试词嵌入OTP检测器性能的基准测试工具。

## 文件说明

- `benchmark_word_embedding.go` - 基准测试源代码
- `benchmark_word_embedding` - 编译后的可执行文件

## 使用方法

### 编译

如果需要重新编译基准测试工具，请在项目根目录执行：

```bash
go build -o test/benchmark/benchmark_word_embedding test/benchmark/benchmark_word_embedding.go
```

### 运行

在项目根目录执行：

```bash
./test/benchmark/benchmark_word_embedding
```

或者进入benchmark目录后执行：

```bash
cd test/benchmark
./benchmark_word_embedding
```

## 测试配置

基准测试的主要配置参数：

- `warmupIterations`: 预热迭代次数 (100)
- `benchmarkRuns`: 基准测试运行次数 (5)
- `iterationsPerRun`: 每次运行的迭代次数 (1000)
- `printProgressStep`: 打印进度的步长 (200)

如需修改这些参数，请编辑`benchmark_word_embedding.go`文件中的相应常量。

## 测试结果

测试结果将直接输出到控制台，包括：

1. 词嵌入检测器的性能结果
2. 平衡增强检测器的性能结果
3. 两种检测器的性能比较

详细的基准测试结果分析请参阅[词嵌入基准测试结果](../../docs/word_embeddings/word_embedding_benchmark_results.md)。 