#!/usr/bin/env python3
"""
验证使用平衡数据集训练的OTP检测模型性能
比较不同模型和平衡方法的效果
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            precision_recall_fscore_support, roc_curve, auc, precision_recall_curve)

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

def evaluate_model(model_path, val_messages, val_labels, model_type, balance_method):
    """评估模型性能"""
    print(f"加载{model_type}模型 ({balance_method}): {model_path}")
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
    
    print(f"\n===== {model_type.upper()}模型性能 ({balance_method}) =====")
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
    
    return y_pred, y_scores, cm, (accuracy, precision, recall, f1)

def plot_confusion_matrix(cm, output_dir, model_type, balance_method):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非OTP', 'OTP'], 
                yticklabels=['非OTP', 'OTP'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_type.upper()}模型混淆矩阵 ({balance_method})')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{model_type}_{balance_method}_confusion_matrix.png')
    plt.savefig(output_path)
    print(f"混淆矩阵图已保存到: {output_path}")

def plot_roc_curve(val_labels, y_scores, output_dir, model_type, balance_method):
    """绘制ROC曲线"""
    # 确保y_scores是一维数组
    if isinstance(y_scores, np.ndarray) and y_scores.ndim > 1:
        y_scores = y_scores[:, 1]
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(val_labels, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_type.upper()}模型ROC曲线 ({balance_method})')
    plt.legend(loc="lower right")
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{model_type}_{balance_method}_roc_curve.png')
    plt.savefig(output_path)
    print(f"ROC曲线已保存到: {output_path}")
    
    return roc_auc

def plot_precision_recall_curve(val_labels, y_scores, output_dir, model_type, balance_method):
    """绘制精确率-召回率曲线"""
    # 确保y_scores是一维数组
    if isinstance(y_scores, np.ndarray) and y_scores.ndim > 1:
        y_scores = y_scores[:, 1]
    
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(val_labels, y_scores)
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{model_type.upper()}模型精确率-召回率曲线 ({balance_method})')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{model_type}_{balance_method}_precision_recall_curve.png')
    plt.savefig(output_path)
    print(f"精确率-召回率曲线已保存到: {output_path}")

def find_optimal_threshold(val_labels, y_scores):
    """寻找最佳决策阈值"""
    # 确保y_scores是一维数组
    if isinstance(y_scores, np.ndarray) and y_scores.ndim > 1:
        y_scores = y_scores[:, 1]
    
    # 计算不同阈值下的精确率和召回率
    precision, recall, thresholds = precision_recall_curve(val_labels, y_scores)
    
    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 找到最大F1分数对应的索引
    optimal_idx = np.argmax(f1_scores)
    
    # 获取最佳阈值
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # 最佳F1分数
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"\n===== 最佳阈值 =====")
    print(f"最佳阈值: {optimal_threshold:.4f}")
    print(f"对应F1分数: {optimal_f1:.4f}")
    print(f"对应精确率: {precision[optimal_idx]:.4f}")
    print(f"对应召回率: {recall[optimal_idx]:.4f}")
    
    return optimal_threshold, optimal_f1

def compare_models(results, output_dir):
    """比较不同模型和平衡方法的性能"""
    # 创建比较表格
    comparison_df = pd.DataFrame(columns=['模型', '平衡方法', '准确率', '精确率', '召回率', 'F1分数', 'AUC'])
    
    # 添加结果
    for model_type, balance_results in results.items():
        for balance_method, (metrics, roc_auc) in balance_results.items():
            accuracy, precision, recall, f1 = metrics
            new_row = {
                '模型': model_type.upper(),
                '平衡方法': balance_method,
                '准确率': accuracy,
                '精确率': precision,
                '召回率': recall,
                'F1分数': f1,
                'AUC': roc_auc
            }
            comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # 按F1分数排序
    comparison_df = comparison_df.sort_values('F1分数', ascending=False)
    
    # 打印比较结果
    print("\n===== 模型比较 =====")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # 保存比较结果
    comparison_path = os.path.join(output_dir, 'models_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"模型比较结果已保存到: {comparison_path}")
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 8))
    
    # 设置分组柱状图
    models = comparison_df['模型'].unique()
    balance_methods = comparison_df['平衡方法'].unique()
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    
    x = np.arange(len(metrics))
    width = 0.15
    
    # 绘制柱状图
    for i, (model, balance) in enumerate(comparison_df[['模型', '平衡方法']].values):
        values = comparison_df.loc[(comparison_df['模型'] == model) & 
                                 (comparison_df['平衡方法'] == balance), 
                                 ['准确率', '精确率', '召回率', 'F1分数']].values[0]
        offset = width * (i - len(comparison_df) / 2 + 0.5)
        plt.bar(x + offset, values, width, label=f'{model} ({balance})')
    
    plt.xlabel('性能指标')
    plt.ylabel('分数')
    plt.title('模型性能比较')
    plt.xticks(x, metrics)
    plt.legend(loc='lower right')
    plt.ylim(0, 1)
    
    # 保存图像
    comparison_plot_path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(comparison_plot_path)
    print(f"模型比较图已保存到: {comparison_plot_path}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证平衡数据集训练的OTP检测模型')
    parser.add_argument('--models_dir', type=str, default=None,
                        help='模型目录路径 (如果不指定，将使用默认路径)')
    parser.add_argument('--data', type=str, default=None,
                        help='验证数据集路径 (如果不指定，将使用默认路径)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录路径 (如果不指定，将使用默认路径)')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    if args.models_dir is None:
        models_dir = os.path.join(base_dir, "models", "go_params")
    else:
        models_dir = args.models_dir
    
    if args.data is None:
        # 使用原始的验证集进行评估，确保公平比较
        data_path = os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    else:
        data_path = args.data
    
    if args.output is None:
        output_dir = os.path.join(base_dir, "results", "validation")
    else:
        output_dir = args.output
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载验证数据
    val_messages, val_labels = load_validation_data(data_path)
    
    # 模型类型和平衡方法
    model_types = ['svm', 'nb', 'rf']
    balance_methods = ['smote', 'undersample']
    
    # 存储结果
    results = {}
    
    # 评估每个模型
    for model_type in model_types:
        results[model_type] = {}
        
        for balance_method in balance_methods:
            model_path = os.path.join(models_dir, f"otp_{model_type}_{balance_method}.joblib")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在: {model_path}")
                continue
            
            # 评估模型
            y_pred, y_scores, cm, metrics = evaluate_model(model_path, val_messages, val_labels, model_type, balance_method)
            
            # 绘制混淆矩阵
            plot_confusion_matrix(cm, output_dir, model_type, balance_method)
            
            # 绘制ROC曲线
            roc_auc = plot_roc_curve(val_labels, y_scores, output_dir, model_type, balance_method)
            
            # 绘制精确率-召回率曲线
            plot_precision_recall_curve(val_labels, y_scores, output_dir, model_type, balance_method)
            
            # 寻找最佳阈值
            optimal_threshold, optimal_f1 = find_optimal_threshold(val_labels, y_scores)
            
            # 存储结果
            results[model_type][balance_method] = (metrics, roc_auc)
    
    # 比较模型
    compare_models(results, output_dir)
    
    print("模型验证完成!")

if __name__ == "__main__":
    main() 