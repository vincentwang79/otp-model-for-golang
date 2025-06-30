#!/usr/bin/env python3
"""
数据集处理脚本 - 清洗和合并OTP和非OTP短信数据集
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv

def process_otp_dataset(file_path):
    """处理OTP数据集"""
    print(f"处理OTP数据集: {file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 只保留短信文本，并添加标签（1表示OTP）
    messages = df['sms_text'].tolist()
    labels = [1] * len(messages)  # 1表示OTP
    
    print(f"  读取了 {len(messages)} 条OTP短信")
    return messages, labels

def process_non_otp_dataset(file_path):
    """处理非OTP数据集（SMSSpamCollection）"""
    print(f"处理非OTP数据集: {file_path}")
    
    messages = []
    labels_ham_spam = []
    
    # 读取SMSSpamCollection文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 格式是"ham/spam\t短信内容"
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"  警告: 跳过格式不正确的行: {line}")
                continue
            
            label, message = parts
            messages.append(message)
            labels_ham_spam.append(label)
    
    # 所有短信都标记为非OTP (0)
    labels = [0] * len(messages)
    
    print(f"  读取了 {len(messages)} 条非OTP短信")
    print(f"  其中 ham (普通短信): {labels_ham_spam.count('ham')}, spam (垃圾短信): {labels_ham_spam.count('spam')}")
    
    return messages, labels

def create_combined_dataset(otp_messages, otp_labels, non_otp_messages, non_otp_labels, test_size=0.1, random_state=42):
    """创建合并数据集并划分训练/验证集"""
    print("创建合并数据集...")
    
    # 为OTP消息创建训练/验证集划分
    otp_train_messages, otp_val_messages, otp_train_labels, otp_val_labels = train_test_split(
        otp_messages, otp_labels, test_size=test_size, random_state=random_state
    )
    
    # 为非OTP消息创建训练/验证集划分
    non_otp_train_messages, non_otp_val_messages, non_otp_train_labels, non_otp_val_labels = train_test_split(
        non_otp_messages, non_otp_labels, test_size=test_size, random_state=random_state
    )
    
    # 合并训练集
    train_messages = otp_train_messages + non_otp_train_messages
    train_labels = otp_train_labels + non_otp_train_labels
    train_splits = ['train'] * len(train_messages)
    
    # 合并验证集
    val_messages = otp_val_messages + non_otp_val_messages
    val_labels = otp_val_labels + non_otp_val_labels
    val_splits = ['val'] * len(val_messages)
    
    # 合并所有数据
    all_messages = train_messages + val_messages
    all_labels = train_labels + val_labels
    all_splits = train_splits + val_splits
    
    # 创建DataFrame
    df = pd.DataFrame({
        'message': all_messages,
        'is_otp': all_labels,
        'split': all_splits
    })
    
    # 随机打乱数据
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"  合并数据集大小: {len(df)}")
    print(f"  训练集: {len(train_messages)}, 验证集: {len(val_messages)}")
    print(f"  OTP消息: {sum(all_labels)}, 非OTP消息: {len(all_labels) - sum(all_labels)}")
    
    return df

def save_dataset(df, output_path):
    """保存数据集到CSV文件"""
    print(f"保存数据集到: {output_path}")
    df.to_csv(output_path, index=False)
    print("  保存完成!")

def generate_stats(df, base_dir):
    """生成数据集统计信息"""
    print("生成数据集统计信息...")
    
    # 计算统计信息
    total_samples = len(df)
    otp_samples = df['is_otp'].sum()
    non_otp_samples = total_samples - otp_samples
    
    train_samples = len(df[df['split'] == 'train'])
    val_samples = len(df[df['split'] == 'val'])
    
    train_otp = len(df[(df['split'] == 'train') & (df['is_otp'] == 1)])
    train_non_otp = train_samples - train_otp
    
    val_otp = len(df[(df['split'] == 'val') & (df['is_otp'] == 1)])
    val_non_otp = val_samples - val_otp
    
    # 创建统计信息文本
    stats = [
        "# 数据集统计信息",
        "",
        f"总样本数: {total_samples}",
        f"OTP短信: {otp_samples} ({otp_samples/total_samples*100:.1f}%)",
        f"非OTP短信: {non_otp_samples} ({non_otp_samples/total_samples*100:.1f}%)",
        "",
        "## 训练集/验证集划分",
        "",
        f"训练集: {train_samples} ({train_samples/total_samples*100:.1f}%)",
        f"  - OTP短信: {train_otp}",
        f"  - 非OTP短信: {train_non_otp}",
        "",
        f"验证集: {val_samples} ({val_samples/total_samples*100:.1f}%)",
        f"  - OTP短信: {val_otp}",
        f"  - 非OTP短信: {val_non_otp}"
    ]
    
    # 保存统计信息
    stats_path = os.path.join(base_dir, "data", "processed", "dataset_stats.md")
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats))
    
    print(f"  统计信息已保存到: {stats_path}")

def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    otp_dataset_path = os.path.join(base_dir, "data", "training-data", "SMS_OTP_10000_samples.csv")
    non_otp_dataset_path = os.path.join(base_dir, "data", "training-data", "SMSSpamCollection")
    output_path = os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 处理数据集
    otp_messages, otp_labels = process_otp_dataset(otp_dataset_path)
    non_otp_messages, non_otp_labels = process_non_otp_dataset(non_otp_dataset_path)
    
    # 创建合并数据集
    df = create_combined_dataset(otp_messages, otp_labels, non_otp_messages, non_otp_labels)
    
    # 保存数据集
    save_dataset(df, output_path)
    
    # 生成数据集统计信息
    generate_stats(df, base_dir)

if __name__ == "__main__":
    main() 