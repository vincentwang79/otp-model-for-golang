#!/usr/bin/env python3
"""
平衡OTP短信数据集，解决类别不平衡问题
支持过采样(SMOTE)和欠采样两种方法
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def load_dataset(dataset_path):
    """加载数据集"""
    print(f"加载数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 分割训练集和验证集
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # 统计信息
    total = len(df)
    otp_count = df['is_otp'].sum()
    non_otp_count = total - otp_count
    
    train_count = len(train_df)
    train_otp = train_df['is_otp'].sum()
    train_non_otp = train_count - train_otp
    
    val_count = len(val_df)
    val_otp = val_df['is_otp'].sum()
    val_non_otp = val_count - val_otp
    
    print(f"总样本数: {total}")
    print(f"OTP短信: {otp_count} ({otp_count/total*100:.1f}%)")
    print(f"非OTP短信: {non_otp_count} ({non_otp_count/total*100:.1f}%)")
    print(f"训练集: {train_count} ({train_count/total*100:.1f}%)")
    print(f"  - OTP短信: {train_otp} ({train_otp/train_count*100:.1f}%)")
    print(f"  - 非OTP短信: {train_non_otp} ({train_non_otp/train_count*100:.1f}%)")
    print(f"验证集: {val_count} ({val_count/total*100:.1f}%)")
    print(f"  - OTP短信: {val_otp} ({val_otp/val_count*100:.1f}%)")
    print(f"  - 非OTP短信: {val_non_otp} ({val_non_otp/val_count*100:.1f}%)")
    
    return df, train_df, val_df

def preprocess_text(text):
    """预处理文本，提取数字模式和关键词"""
    # 检查是否为字符串
    if not isinstance(text, str):
        text = str(text)
    
    # 提取数字模式
    digits = re.findall(r'\d+', text)
    digit_pattern = " ".join([f"DIGITS_{len(d)}" for d in digits])
    
    # 提取关键词 - 扩展关键词列表
    otp_keywords = [
        "验证码", "code", "验证", "verification", "otp", "password", "密码", "有效期", "valid",
        "security", "安全", "confirm", "确认", "login", "登录", "access", "账户", "account",
        "pin", "认证", "authentication", "verify", "secure", "保密", "confidential"
    ]
    keyword_pattern = ""
    for keyword in otp_keywords:
        if keyword.lower() in text.lower():
            keyword_pattern += f" KEYWORD_{keyword.lower()}"
    
    # 结合原始文本和提取的模式
    return f"{text} {digit_pattern} {keyword_pattern}"

def extract_features(messages):
    """提取文本特征"""
    print("提取文本特征...")
    # 预处理文本
    processed_messages = [preprocess_text(msg) for msg in messages]
    
    # 创建TF-IDF特征
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=3000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(processed_messages)
    return X, vectorizer

def balance_with_smote(X_train, y_train, sampling_strategy=0.8):
    """使用SMOTE过采样平衡数据集"""
    print(f"使用SMOTE过采样，目标比例: {sampling_strategy}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # 统计信息
    total = len(y_resampled)
    otp_count = sum(y_resampled)
    non_otp_count = total - otp_count
    
    print(f"过采样后总样本数: {total}")
    print(f"OTP短信: {otp_count} ({otp_count/total*100:.1f}%)")
    print(f"非OTP短信: {non_otp_count} ({non_otp_count/total*100:.1f}%)")
    
    return X_resampled, y_resampled

def balance_with_undersampling(X_train, y_train, sampling_strategy=0.8):
    """使用欠采样平衡数据集"""
    print(f"使用欠采样，目标比例: {sampling_strategy}")
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    # 统计信息
    total = len(y_resampled)
    otp_count = sum(y_resampled)
    non_otp_count = total - otp_count
    
    print(f"欠采样后总样本数: {total}")
    print(f"OTP短信: {otp_count} ({otp_count/total*100:.1f}%)")
    print(f"非OTP短信: {non_otp_count} ({non_otp_count/total*100:.1f}%)")
    
    return X_resampled, y_resampled

def create_balanced_dataset(train_df, val_df, X_resampled, y_resampled, method):
    """创建平衡后的数据集"""
    print("创建平衡后的数据集...")
    
    # 获取原始训练集的索引
    original_train_indices = train_df.index.tolist()
    
    # 创建新的训练集DataFrame
    resampled_indices = []
    new_messages = []
    
    # 处理过采样/欠采样后的数据
    for i, (is_otp) in enumerate(y_resampled):
        if i < len(original_train_indices):
            # 原始样本
            resampled_indices.append(original_train_indices[i])
        else:
            # 合成样本 (SMOTE) 或 保留的欠采样样本
            # 这里我们不能直接使用索引，因为SMOTE创建了新样本
            # 对于欠采样，我们只保留了部分原始样本
            # 所以我们需要创建新的消息
            if method == 'smote':
                # 为SMOTE合成的样本创建一个标记
                msg = f"[SYNTHETIC_{i}] OTP样本"
                new_messages.append((msg, 1))
            else:
                # 欠采样不需要创建新消息，我们只需跳过被移除的样本
                pass
    
    # 创建新的训练集
    if method == 'smote':
        # 对于SMOTE，我们需要添加合成样本
        train_resampled = train_df.copy()
        
        # 添加合成样本
        for msg, is_otp in new_messages:
            new_row = pd.DataFrame({
                'message': [msg],
                'is_otp': [is_otp],
                'split': ['train']
            })
            train_resampled = pd.concat([train_resampled, new_row], ignore_index=True)
    else:
        # 对于欠采样，我们需要保留对应的样本
        # 创建一个布尔掩码，表示哪些样本被保留
        mask = np.zeros(len(train_df), dtype=bool)
        mask[np.arange(len(y_resampled))] = True
        
        # 使用掩码筛选样本
        train_resampled = train_df[mask].copy()
    
    # 合并训练集和验证集
    balanced_df = pd.concat([train_resampled, val_df], ignore_index=True)
    
    # 统计信息
    total = len(balanced_df)
    otp_count = balanced_df['is_otp'].sum()
    non_otp_count = total - otp_count
    
    train_count = len(train_resampled)
    train_otp = train_resampled['is_otp'].sum()
    train_non_otp = train_count - train_otp
    
    val_count = len(val_df)
    
    print(f"平衡后总样本数: {total}")
    print(f"OTP短信: {otp_count} ({otp_count/total*100:.1f}%)")
    print(f"非OTP短信: {non_otp_count} ({non_otp_count/total*100:.1f}%)")
    print(f"训练集: {train_count} ({train_count/total*100:.1f}%)")
    print(f"  - OTP短信: {train_otp} ({train_otp/train_count*100:.1f}%)")
    print(f"  - 非OTP短信: {train_non_otp} ({train_non_otp/train_count*100:.1f}%)")
    
    return balanced_df

def save_dataset(df, output_path):
    """保存数据集"""
    print(f"保存平衡后的数据集: {output_path}")
    df.to_csv(output_path, index=False)
    print("保存完成!")

def generate_stats(df, output_path):
    """生成数据集统计信息"""
    print("生成数据集统计信息...")
    
    # 计算统计信息
    total = len(df)
    otp_count = df['is_otp'].sum()
    non_otp_count = total - otp_count
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    train_count = len(train_df)
    train_otp = train_df['is_otp'].sum()
    train_non_otp = train_count - train_otp
    
    val_count = len(val_df)
    val_otp = val_df['is_otp'].sum()
    val_non_otp = val_count - val_otp
    
    # 创建统计信息文本
    stats = [
        "# 平衡后的数据集统计信息",
        "",
        f"总样本数: {total}",
        f"OTP短信: {otp_count} ({otp_count/total*100:.1f}%)",
        f"非OTP短信: {non_otp_count} ({non_otp_count/total*100:.1f}%)",
        "",
        "## 训练集/验证集划分",
        "",
        f"训练集: {train_count} ({train_count/total*100:.1f}%)",
        f"  - OTP短信: {train_otp} ({train_otp/train_count*100:.1f}%)",
        f"  - 非OTP短信: {train_non_otp} ({train_non_otp/train_count*100:.1f}%)",
        "",
        f"验证集: {val_count} ({val_count/total*100:.1f}%)",
        f"  - OTP短信: {val_otp} ({val_otp/val_count*100:.1f}%)",
        f"  - 非OTP短信: {val_non_otp} ({val_non_otp/val_count*100:.1f}%)"
    ]
    
    # 保存统计信息
    with open(output_path, 'w') as f:
        f.write('\n'.join(stats))
    
    print(f"统计信息已保存到: {output_path}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='平衡OTP短信数据集')
    parser.add_argument('--method', type=str, default='smote', choices=['smote', 'undersample'],
                        help='平衡方法: smote (过采样), undersample (欠采样)')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='目标OTP:非OTP比例 (0.8表示OTP:非OTP = 1:1.25)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入数据集路径 (如果不指定，将使用默认路径)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出数据集路径 (如果不指定，将使用默认路径)')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    if args.input is None:
        input_path = os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    else:
        input_path = args.input
    
    if args.output is None:
        if args.method == 'smote':
            output_path = os.path.join(base_dir, "data", "processed", "balanced_smote_dataset.csv")
            stats_path = os.path.join(base_dir, "data", "processed", "balanced_smote_stats.md")
        else:
            output_path = os.path.join(base_dir, "data", "processed", "balanced_undersampled_dataset.csv")
            stats_path = os.path.join(base_dir, "data", "processed", "balanced_undersampled_stats.md")
    else:
        output_path = args.output
        stats_path = os.path.join(os.path.dirname(args.output), "balanced_stats.md")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载数据集
    df, train_df, val_df = load_dataset(input_path)
    
    # 提取特征
    X_train, vectorizer = extract_features(train_df['message'])
    y_train = train_df['is_otp'].values
    
    # 平衡数据集
    if args.method == 'smote':
        X_resampled, y_resampled = balance_with_smote(X_train, y_train, args.ratio)
    else:
        X_resampled, y_resampled = balance_with_undersampling(X_train, y_train, args.ratio)
    
    # 创建平衡后的数据集
    balanced_df = create_balanced_dataset(train_df, val_df, X_resampled, y_resampled, args.method)
    
    # 保存数据集
    save_dataset(balanced_df, output_path)
    
    # 生成统计信息
    generate_stats(balanced_df, stats_path)
    
    print("数据集平衡完成!")

if __name__ == "__main__":
    main() 