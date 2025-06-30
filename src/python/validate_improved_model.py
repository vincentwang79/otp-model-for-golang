#!/usr/bin/env python3
"""
使用验证集数据评估改进的OTP检测模型的性能
支持验证不同类型的模型：SVM、朴素贝叶斯、随机森林
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def load_validation_data(csv_file):
    """从CSV文件加载验证集数据"""
    print(f"加载CSV数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 只使用验证集数据
    val_df = df[df['split'] == 'val']
    
    # 获取验证集的消息和标签
    val_messages = val_df['message'].tolist()
    val_labels = val_df['is_otp'].tolist()
    
    # 统计验证集中OTP和非OTP消息数量
    val_otp_count = sum(val_labels)
    val_non_otp_count = len(val_labels) - val_otp_count
    
    print(f"验证集大小: {len(val_df)}")
    print(f"验证集OTP消息: {val_otp_count}, 非OTP消息: {val_non_otp_count}")
    
    return val_messages, val_labels

def evaluate_model(model_path, val_messages, val_labels, model_type):
    """评估模型性能"""
    print(f"加载{model_type}模型: {model_path}")
    model = joblib.load(model_path)
    
    # 预测
    print("进行预测...")
    y_pred = model.predict(val_messages)
    
    # 计算置信度分数（如果模型支持）
    try:
        y_scores = model.decision_function(val_messages)
    except (AttributeError, NotImplementedError):
        try:
            # 对于不支持decision_function的模型，尝试使用predict_proba
            y_scores = model.predict_proba(val_messages)[:, 1]
        except (AttributeError, NotImplementedError):
            # 如果都不支持，使用预测结果作为分数
            y_scores = y_pred
    
    # 评估性能
    accuracy = accuracy_score(val_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average='binary')
    
    print(f"\n===== {model_type.upper()}模型性能 =====")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 详细分类报告
    print("\n===== 分类报告 =====")
    print(classification_report(val_labels, y_pred, target_names=['非OTP', 'OTP']))
    
    # 混淆矩阵
    cm = confusion_matrix(val_labels, y_pred)
    print("\n===== 混淆矩阵 =====")
    print(cm)
    
    return y_pred, y_scores, cm

def plot_confusion_matrix(cm, output_dir, model_type):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非OTP', 'OTP'], 
                yticklabels=['非OTP', 'OTP'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_type.upper()}模型混淆矩阵')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{model_type}_confusion_matrix.png')
    plt.savefig(output_path)
    print(f"混淆矩阵图已保存到: {output_path}")

def analyze_errors(val_messages, val_labels, y_pred, y_scores, output_dir, model_type):
    """分析错误预测"""
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'message': val_messages,
        'true_label': val_labels,
        'predicted': y_pred,
        'score': y_scores
    })
    
    # 找出错误预测
    false_positives = results_df[(results_df['true_label'] == 0) & (results_df['predicted'] == 1)]
    false_negatives = results_df[(results_df['true_label'] == 1) & (results_df['predicted'] == 0)]
    
    print("\n===== 错误分析 =====")
    print(f"假阳性 (非OTP被预测为OTP): {len(false_positives)}")
    print(f"假阴性 (OTP被预测为非OTP): {len(false_negatives)}")
    
    # 保存错误预测
    fp_path = os.path.join(output_dir, f'{model_type}_false_positives.csv')
    fn_path = os.path.join(output_dir, f'{model_type}_false_negatives.csv')
    
    false_positives.to_csv(fp_path, index=False)
    false_negatives.to_csv(fn_path, index=False)
    
    print(f"假阳性样本已保存到: {fp_path}")
    print(f"假阴性样本已保存到: {fn_path}")
    
    # 保存所有预测结果
    all_results_path = os.path.join(output_dir, f'{model_type}_validation_results.csv')
    results_df.to_csv(all_results_path, index=False)
    print(f"所有预测结果已保存到: {all_results_path}")

def compare_with_original(original_model_path, improved_model_path, val_messages, val_labels, output_dir, model_type):
    """比较改进模型和原始模型的性能"""
    print(f"\n===== 模型比较 =====")
    print(f"加载原始SVM模型: {original_model_path}")
    original_model = joblib.load(original_model_path)
    
    print(f"加载改进{model_type.upper()}模型: {improved_model_path}")
    improved_model = joblib.load(improved_model_path)
    
    # 预测
    print("进行预测...")
    original_pred = original_model.predict(val_messages)
    improved_pred = improved_model.predict(val_messages)
    
    # 计算性能指标
    original_accuracy = accuracy_score(val_labels, original_pred)
    improved_accuracy = accuracy_score(val_labels, improved_pred)
    
    original_precision, original_recall, original_f1, _ = precision_recall_fscore_support(val_labels, original_pred, average='binary')
    improved_precision, improved_recall, improved_f1, _ = precision_recall_fscore_support(val_labels, improved_pred, average='binary')
    
    # 创建比较表格
    comparison_df = pd.DataFrame({
        '指标': ['准确率', '精确率', '召回率', 'F1分数'],
        '原始SVM模型': [original_accuracy, original_precision, original_recall, original_f1],
        f'改进{model_type.upper()}模型': [improved_accuracy, improved_precision, improved_recall, improved_f1],
        '提升': [
            improved_accuracy - original_accuracy,
            improved_precision - original_precision,
            improved_recall - original_recall,
            improved_f1 - original_f1
        ]
    })
    
    # 打印比较结果
    print("\n模型性能比较:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # 保存比较结果
    comparison_path = os.path.join(output_dir, f'{model_type}_vs_original_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"模型比较结果已保存到: {comparison_path}")
    
    # 绘制性能比较图
    plt.figure(figsize=(10, 6))
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['原始SVM模型'], width, label='原始SVM模型')
    plt.bar(x + width/2, comparison_df[f'改进{model_type.upper()}模型'], width, label=f'改进{model_type.upper()}模型')
    
    plt.xlabel('性能指标')
    plt.ylabel('分数')
    plt.title('模型性能比较')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    # 保存图像
    comparison_plot_path = os.path.join(output_dir, f'{model_type}_vs_original_comparison.png')
    plt.savefig(comparison_plot_path)
    print(f"模型比较图已保存到: {comparison_plot_path}")

def compare_all_models(val_messages, val_labels, models_dict, output_dir):
    """比较所有可用的模型"""
    if len(models_dict) <= 1:
        print("没有足够的模型可供比较")
        return
    
    print(f"\n===== 所有模型比较 =====")
    
    # 存储所有模型的性能指标
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    model_names = []
    
    # 对每个模型进行评估
    for model_name, model_path in models_dict.items():
        if not os.path.exists(model_path):
            continue
            
        print(f"评估{model_name}模型...")
        model = joblib.load(model_path)
        y_pred = model.predict(val_messages)
        
        # 计算性能指标
        accuracy = accuracy_score(val_labels, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average='binary')
        
        # 存储结果
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        model_names.append(model_name)
    
    # 创建比较表格
    comparison_df = pd.DataFrame({
        '模型': model_names,
        '准确率': accuracies,
        '精确率': precisions,
        '召回率': recalls,
        'F1分数': f1_scores
    })
    
    # 按F1分数排序
    comparison_df = comparison_df.sort_values('F1分数', ascending=False)
    
    # 打印比较结果
    print("\n所有模型性能比较:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # 保存比较结果
    comparison_path = os.path.join(output_dir, 'all_models_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"所有模型比较结果已保存到: {comparison_path}")
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 8))
    
    # 设置分组柱状图
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    x = np.arange(len(model_names))
    width = 0.2
    
    # 绘制每个指标的柱状图
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, comparison_df[metric], width, label=metric)
    
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.title('所有模型性能比较')
    plt.xticks(x, comparison_df['模型'])
    plt.legend()
    plt.ylim(0, 1)
    
    # 保存图像
    comparison_plot_path = os.path.join(output_dir, 'all_models_comparison.png')
    plt.savefig(comparison_plot_path)
    print(f"所有模型比较图已保存到: {comparison_plot_path}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证OTP检测模型')
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'nb', 'rf', 'all'],
                        help='指定要验证的模型类型: svm (支持向量机), nb (朴素贝叶斯), rf (随机森林), all (所有模型)')
    parser.add_argument('--data', type=str, default=None,
                        help='CSV数据文件路径 (如果不指定，将使用默认路径)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录路径 (如果不指定，将使用默认路径)')
    parser.add_argument('--compare', action='store_true',
                        help='是否与原始SVM模型进行比较')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    model_type = args.model
    compare_flag = args.compare
    
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 数据路径
    csv_file = args.data if args.data else os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    
    # 输出目录
    output_dir = args.output if args.output else os.path.join(base_dir, "results", "validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型路径
    original_model_path = os.path.join(base_dir, "models", "go_params", "otp_svm.joblib")
    
    # 定义所有可能的模型路径
    models_dict = {
        'svm': os.path.join(base_dir, "models", "go_params", "otp_svm_improved.joblib"),
        'nb': os.path.join(base_dir, "models", "go_params", "otp_nb_improved.joblib"),
        'rf': os.path.join(base_dir, "models", "go_params", "otp_rf_improved.joblib")
    }
    
    # 加载验证数据
    val_messages, val_labels = load_validation_data(csv_file)
    
    # 根据指定的模型类型进行验证
    if model_type == 'all':
        # 验证所有可用的模型
        compare_all_models(val_messages, val_labels, models_dict, output_dir)
        
        # 单独验证每个可用的模型
        for model_name, model_path in models_dict.items():
            if os.path.exists(model_path):
                print(f"\n===== 验证{model_name.upper()}模型 =====")
                y_pred, y_scores, cm = evaluate_model(model_path, val_messages, val_labels, model_name)
                plot_confusion_matrix(cm, output_dir, model_name)
                analyze_errors(val_messages, val_labels, y_pred, y_scores, output_dir, model_name)
                
                if compare_flag and os.path.exists(original_model_path):
                    compare_with_original(original_model_path, model_path, val_messages, val_labels, output_dir, model_name)
    else:
        # 验证指定的模型
        model_path = models_dict.get(model_type)
        
        if not os.path.exists(model_path):
            print(f"错误: 找不到{model_type}模型文件: {model_path}")
            sys.exit(1)
        
        y_pred, y_scores, cm = evaluate_model(model_path, val_messages, val_labels, model_type)
        plot_confusion_matrix(cm, output_dir, model_type)
        analyze_errors(val_messages, val_labels, y_pred, y_scores, output_dir, model_type)
        
        if compare_flag and os.path.exists(original_model_path):
            compare_with_original(original_model_path, model_path, val_messages, val_labels, output_dir, model_type)
    
    print(f"\n验证完成!")

if __name__ == "__main__":
    main() 