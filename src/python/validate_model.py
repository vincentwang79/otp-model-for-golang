#!/usr/bin/env python3
"""
使用验证集数据评估OTP检测模型的性能
"""

import os
import sys
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

def evaluate_model(model_path, val_messages, val_labels):
    """评估模型性能"""
    print(f"加载模型: {model_path}")
    model = joblib.load(model_path)
    
    # 预测
    print("进行预测...")
    y_pred = model.predict(val_messages)
    
    # 计算置信度分数
    y_scores = model.decision_function(val_messages)
    
    # 评估性能
    accuracy = accuracy_score(val_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average='binary')
    
    print("\n===== 模型性能 =====")
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

def plot_confusion_matrix(cm, output_dir):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非OTP', 'OTP'], 
                yticklabels=['非OTP', 'OTP'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 保存图像
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    print(f"混淆矩阵图已保存到: {output_path}")

def analyze_errors(val_messages, val_labels, y_pred, y_scores, output_dir):
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
    fp_path = os.path.join(output_dir, 'false_positives.csv')
    fn_path = os.path.join(output_dir, 'false_negatives.csv')
    
    false_positives.to_csv(fp_path, index=False)
    false_negatives.to_csv(fn_path, index=False)
    
    print(f"假阳性样本已保存到: {fp_path}")
    print(f"假阴性样本已保存到: {fn_path}")
    
    # 保存所有预测结果
    all_results_path = os.path.join(output_dir, 'validation_results.csv')
    results_df.to_csv(all_results_path, index=False)
    print(f"所有预测结果已保存到: {all_results_path}")

def main():
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 数据路径
    csv_file = os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    
    # 模型路径
    model_path = os.path.join(base_dir, "models", "go_params", "otp_svm.joblib")
    
    # 输出目录
    output_dir = os.path.join(base_dir, "results", "validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载验证数据
    val_messages, val_labels = load_validation_data(csv_file)
    
    # 评估模型
    y_pred, y_scores, cm = evaluate_model(model_path, val_messages, val_labels)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, output_dir)
    
    # 分析错误
    analyze_errors(val_messages, val_labels, y_pred, y_scores, output_dir)
    
    print("\n验证完成!")

if __name__ == "__main__":
    main() 