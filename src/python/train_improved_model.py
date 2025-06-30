#!/usr/bin/env python3
"""
训练改进的OTP检测模型，解决验证集上的低召回率问题
支持多种模型类型：SVM、朴素贝叶斯、随机森林
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import json
import re
from sklearn.metrics import classification_report, confusion_matrix

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
    
    # 分割训练集和验证集
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # 获取训练集的消息和标签
    train_messages = train_df['message'].tolist()
    train_labels = train_df['is_otp'].tolist()
    
    # 获取验证集的消息和标签
    val_messages = val_df['message'].tolist()
    val_labels = val_df['is_otp'].tolist()
    
    # 统计训练集中OTP和非OTP消息数量
    train_otp_count = sum(train_labels)
    train_non_otp_count = len(train_labels) - train_otp_count
    
    # 统计验证集中OTP和非OTP消息数量
    val_otp_count = sum(val_labels)
    val_non_otp_count = len(val_labels) - val_otp_count
    
    print(f"总数据集大小: {len(df)}")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"训练集OTP消息: {train_otp_count}, 非OTP消息: {train_non_otp_count}")
    print(f"验证集OTP消息: {val_otp_count}, 非OTP消息: {val_non_otp_count}")
    
    return train_messages, train_labels, val_messages, val_labels

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

def create_model(model_type):
    """根据指定的模型类型创建相应的模型"""
    # 创建TF-IDF向量化器 (比CountVectorizer更好的特征表示)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=3000,  # 增加特征数量
        min_df=2,  # 降低最小文档频率
        max_df=0.9,  # 提高最大文档频率
        ngram_range=(1, 3),  # 使用1-gram、2-gram和3-gram
        sublinear_tf=True  # 使用次线性TF缩放
    )
    
    if model_type == 'svm':
        # 创建线性SVM分类器，大幅增加OTP类别的权重
        classifier = LinearSVC(
            C=10.0,  # 增加正则化参数
            class_weight={0: 1, 1: 20},  # 大幅增加OTP类别的权重
            dual=False,
            max_iter=10000
        )
        print("使用SVM模型")
    elif model_type == 'nb':
        # 创建朴素贝叶斯分类器
        classifier = MultinomialNB(
            alpha=0.1,  # 平滑参数
            fit_prior=True,
            class_prior=[0.4, 0.6]  # 类别先验概率，偏向OTP类
        )
        print("使用朴素贝叶斯模型")
    elif model_type == 'rf':
        # 创建随机森林分类器
        classifier = RandomForestClassifier(
            n_estimators=200,  # 树的数量
            max_depth=None,  # 树的最大深度
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight={0: 1, 1: 20},  # 类别权重
            random_state=42,
            n_jobs=-1  # 使用所有可用的CPU核心
        )
        print("使用随机森林模型")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建Pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])
    
    return pipeline

def train_improved_model(train_messages, train_labels, val_messages, val_labels, model_type):
    """训练改进的OTP检测模型"""
    # 预处理文本
    print("预处理文本...")
    processed_train_messages = [preprocess_text(msg) for msg in train_messages]
    processed_val_messages = [preprocess_text(msg) for msg in val_messages]
    
    # 创建模型
    pipeline = create_model(model_type)
    
    # 训练模型
    print("训练模型...")
    pipeline.fit(processed_train_messages, train_labels)
    
    # 在验证集上评估模型
    print("在验证集上评估模型...")
    val_pred = pipeline.predict(processed_val_messages)
    
    # 打印分类报告
    print("\n===== 验证集分类报告 =====")
    print(classification_report(val_labels, val_pred, target_names=['非OTP', 'OTP']))
    
    # 打印混淆矩阵
    cm = confusion_matrix(val_labels, val_pred)
    print("\n===== 验证集混淆矩阵 =====")
    print(cm)
    
    return pipeline, processed_train_messages

def export_model_params(pipeline, output_dir, processed_messages, model_type):
    """导出模型参数为JSON格式"""
    # 获取向量化器
    vectorizer = pipeline.named_steps['vectorizer']
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    
    # 获取分类器
    classifier = pipeline.named_steps['classifier']
    
    # 创建基本模型参数字典
    model_params = {
        "vocabulary": vocabulary,
        "binary": False,  # TF-IDF不是二进制的
        "lowercase": vectorizer.lowercase,
        "model_type": model_type,
        "otp_keywords": [
            "验证码", "code", "验证", "verification", "otp", "password", "密码", "有效期", "valid",
            "security", "安全", "confirm", "确认", "login", "登录", "access", "账户", "account",
            "pin", "认证", "authentication", "verify", "secure", "保密", "confidential"
        ]
    }
    
    # 根据模型类型添加特定参数
    if model_type == 'svm':
        model_params["coef"] = classifier.coef_[0].tolist()
        model_params["intercept"] = float(classifier.intercept_[0])
    elif model_type == 'nb':
        model_params["feature_log_prob"] = classifier.feature_log_prob_.tolist()
        model_params["class_log_prior"] = classifier.class_log_prior_.tolist()
    elif model_type == 'rf':
        # 随机森林模型太复杂，无法直接导出为JSON
        # 只能保存完整模型，不导出参数
        pass
    
    # 保存为JSON
    params_path = os.path.join(output_dir, f"otp_{model_type}_improved_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(model_params, f, indent=2, cls=NumpyEncoder)
    
    print(f"改进的模型参数已保存到: {params_path}")
    
    # 保存完整模型
    model_path = os.path.join(output_dir, f"otp_{model_type}_improved.joblib")
    joblib.dump(pipeline, model_path)
    print(f"改进的完整模型已保存到: {model_path}")
    
    # 保存TFIDF参数
    tfidf_params = {
        "vocabulary": vocabulary,
        "binary": False,
        "lowercase": vectorizer.lowercase,
        "sublinear_tf": True
    }
    tfidf_path = os.path.join(output_dir, f"otp_tfidf_{model_type}_improved_params.json")
    with open(tfidf_path, 'w', encoding='utf-8') as f:
        json.dump(tfidf_params, f, indent=2, cls=NumpyEncoder)
    
    print(f"改进的TFIDF参数已保存到: {tfidf_path}")
    
    return params_path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练改进的OTP检测模型')
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'nb', 'rf'],
                        help='指定模型类型: svm (支持向量机), nb (朴素贝叶斯), rf (随机森林)')
    parser.add_argument('--data', type=str, default=None,
                        help='CSV数据文件路径 (如果不指定，将使用默认路径)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录路径 (如果不指定，将使用默认路径)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    model_type = args.model
    
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 数据路径
    csv_file = args.data if args.data else os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    
    # 输出路径
    output_dir = args.output if args.output else os.path.join(base_dir, "models", "go_params")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    train_messages, train_labels, val_messages, val_labels = load_data_from_csv(csv_file)
    
    # 训练改进的模型
    pipeline, processed_messages = train_improved_model(train_messages, train_labels, val_messages, val_labels, model_type)
    
    # 导出模型参数
    params_path = export_model_params(pipeline, output_dir, processed_messages, model_type)
    
    print(f"\n改进的{model_type}模型训练完成!")

if __name__ == "__main__":
    main() 