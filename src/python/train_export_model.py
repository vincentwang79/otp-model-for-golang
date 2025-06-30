#!/usr/bin/env python3
"""
将OTP检测模型转换为简单格式，以便在Go中使用
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import json
import re

# 自定义JSON编码器，处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_data_from_csv(csv_file):
    """从CSV文件加载数据"""
    print(f"加载CSV数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 只使用训练集数据
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # 获取训练集的消息和标签
    train_messages = train_df['message'].tolist()
    train_labels = train_df['is_otp'].tolist()
    
    # 统计训练集中OTP和非OTP消息数量
    train_otp_count = sum(train_labels)
    train_non_otp_count = len(train_labels) - train_otp_count
    
    print(f"总数据集大小: {len(df)}")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"训练集OTP消息: {train_otp_count}, 非OTP消息: {train_non_otp_count}")
    
    return train_messages, train_labels

def preprocess_text(text):
    """预处理文本，提取数字模式和关键词"""
    # 提取数字模式
    digits = re.findall(r'\d+', text)
    digit_pattern = " ".join([f"DIGITS_{len(d)}" for d in digits])
    
    # 提取关键词
    otp_keywords = ["验证码", "code", "验证", "verification", "otp", "password", "密码", "有效期", "valid"]
    keyword_pattern = ""
    for keyword in otp_keywords:
        if keyword.lower() in text.lower():
            keyword_pattern += f" KEYWORD_{keyword.lower()}"
    
    # 结合原始文本和提取的模式
    return f"{text} {digit_pattern} {keyword_pattern}"

def train_model(messages, labels):
    """训练OTP检测模型"""
    # 预处理文本
    processed_messages = [preprocess_text(msg) for msg in messages]
    
    # 创建CountVectorizer
    vectorizer = CountVectorizer(
        lowercase=True,
        max_features=2000,  # 增加特征数量
        min_df=2,  # 降低最小文档频率
        max_df=0.9,  # 提高最大文档频率
        ngram_range=(1, 2),  # 使用1-gram和2-gram
        binary=True  # 二进制特征
    )
    
    # 创建线性SVM分类器
    svm = LinearSVC(
        C=1.0,
        class_weight={0: 1, 1: 5},  # 增加OTP类别的权重
        dual=False,
        max_iter=10000
    )
    
    # 创建Pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", svm)
    ])
    
    # 训练模型
    print("训练模型...")
    pipeline.fit(processed_messages, labels)
    
    return pipeline, processed_messages

def export_model_params(pipeline, output_dir, processed_messages):
    """导出模型参数为JSON格式"""
    # 获取向量化器
    vectorizer = pipeline.named_steps['vectorizer']
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    
    # 获取分类器
    classifier = pipeline.named_steps['classifier']
    coef = classifier.coef_[0].tolist()
    intercept = float(classifier.intercept_[0])
    
    # 创建模型参数字典
    model_params = {
        "vocabulary": vocabulary,
        "coef": coef,
        "intercept": intercept,
        "binary": vectorizer.binary,
        "lowercase": vectorizer.lowercase,
        "otp_keywords": ["验证码", "code", "验证", "verification", "otp", "password", "密码", "有效期", "valid"]
    }
    
    # 保存为JSON
    params_path = os.path.join(output_dir, "otp_svm_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(model_params, f, indent=2, cls=NumpyEncoder)
    
    print(f"模型参数已保存到: {params_path}")
    
    # 保存完整模型
    model_path = os.path.join(output_dir, "otp_svm.joblib")
    joblib.dump(pipeline, model_path)
    print(f"完整模型已保存到: {model_path}")
    
    # 保存一些示例处理后的文本，用于调试
    examples_path = os.path.join(output_dir, "processed_examples.txt")
    with open(examples_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(processed_messages[:20]):
            f.write(f"示例 {i+1}:\n{text}\n\n")
    
    print(f"处理后的示例文本已保存到: {examples_path}")
    
    # 保存TFIDF参数，用于可能的后续处理
    tfidf_params = {
        "vocabulary": vocabulary,
        "binary": vectorizer.binary,
        "lowercase": vectorizer.lowercase
    }
    tfidf_path = os.path.join(output_dir, "otp_tfidf_params.json")
    with open(tfidf_path, 'w', encoding='utf-8') as f:
        json.dump(tfidf_params, f, indent=2, cls=NumpyEncoder)
    
    print(f"TFIDF参数已保存到: {tfidf_path}")
    
    return params_path

def main():
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    # 数据路径 - 使用combined_sms_dataset.csv数据集
    csv_file = os.path.join(current_dir, "..", "..", "data", "processed", "combined_sms_dataset.csv")
    
    # 输出路径
    output_dir = os.path.join(current_dir, "..", "..", "models", "go_params")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    messages, labels = load_data_from_csv(csv_file)
    
    # 训练模型
    pipeline, processed_messages = train_model(messages, labels)
    
    # 导出模型参数
    params_path = export_model_params(pipeline, output_dir, processed_messages)
    
    print("转换完成!")

if __name__ == "__main__":
    main() 