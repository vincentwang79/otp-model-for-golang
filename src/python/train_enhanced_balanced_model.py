#!/usr/bin/env python3
"""
结合增强特征工程和数据平衡策略训练OTP检测模型
支持SMOTE过采样和欠采样两种平衡方法
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import json
import matplotlib.pyplot as plt
import time

# 导入增强特征工程模块
from enhanced_feature_engineering import preprocess_text_for_training, create_enhanced_vectorizer, OTP_KEYWORDS

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
    
    # 获取验证集的消息和标签
    val_messages = val_df['message'].tolist()
    val_labels = val_df['is_otp'].tolist()
    
    # 统计训练集中OTP和非OTP消息数量
    train_otp_count = sum(train_labels)
    train_non_otp_count = len(train_labels) - train_otp_count
    
    print(f"总数据集大小: {len(df)}")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"训练集OTP消息: {train_otp_count}, 非OTP消息: {train_non_otp_count}")
    print(f"训练集类别比例: OTP {train_otp_count/len(train_labels):.2f}, 非OTP {train_non_otp_count/len(train_labels):.2f}")
    
    return train_messages, train_labels, val_messages, val_labels

def balance_dataset(X_train, y_train, balance_method='smote', target_ratio=1.0):
    """平衡数据集
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        balance_method: 平衡方法 ('smote' 或 'undersample')
        target_ratio: 目标正负样本比例 (正样本数/负样本数)
    
    返回:
        X_resampled: 平衡后的特征
        y_resampled: 平衡后的标签
    """
    print(f"使用{balance_method}方法平衡数据集...")
    
    # 计算当前类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"原始类别分布: {class_dist}")
    
    # 根据目标比例计算采样策略
    if 0 in class_dist and 1 in class_dist:
        neg_count = class_dist[0]
        pos_count = class_dist[1]
        
        if balance_method == 'smote':
            # SMOTE过采样 - 增加少数类样本
            # 计算采样策略: 少数类样本数 / 多数类样本数
            if pos_count < neg_count:
                # OTP是少数类
                sampling_strategy = min(target_ratio, 0.9)  # 限制最大比例为0.9，避免过拟合
            else:
                # 非OTP是少数类
                sampling_strategy = max(1/target_ratio, 0.1)  # 限制最小比例为0.1
                
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        else:
            # 欠采样 - 减少多数类样本
            if pos_count > neg_count:
                # OTP是多数类
                sampling_strategy = min(1/target_ratio, 0.9)  # 限制最大比例为0.9
            else:
                # 非OTP是多数类
                sampling_strategy = max(target_ratio, 0.1)  # 限制最小比例为0.1
                
            resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        
        # 执行重采样
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        # 打印重采样后的类别分布
        unique, counts = np.unique(y_resampled, return_counts=True)
        resampled_dist = dict(zip(unique, counts))
        print(f"重采样后类别分布: {resampled_dist}")
        
        return X_resampled, y_resampled
    else:
        print("警告: 数据集中缺少某个类别，无法进行平衡")
        return X_train, y_train

def train_model(messages, labels, model_type='svm', balance_method='smote'):
    """训练OTP检测模型"""
    start_time = time.time()
    
    # 预处理文本，使用增强特征
    print("预处理文本，提取增强特征...")
    processed_messages = preprocess_text_for_training(messages)
    
    # 创建增强的向量化器
    vectorizer = create_enhanced_vectorizer(max_features=3000, ngram_range=(1, 3))
    
    # 特征提取
    X = vectorizer.fit_transform(processed_messages)
    
    # 平衡数据集
    X_balanced, y_balanced = balance_dataset(X, labels, balance_method, target_ratio=1.0)
    
    # 根据模型类型创建分类器
    if model_type == 'svm':
        # 线性SVM分类器
        classifier = LinearSVC(
            C=1.0,
            class_weight='balanced',  # 自动平衡类别权重
            dual=False,
            max_iter=10000
        )
    elif model_type == 'nb':
        # 朴素贝叶斯分类器
        classifier = MultinomialNB(alpha=0.1)
    elif model_type == 'rf':
        # 随机森林分类器
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    print(f"训练{model_type.upper()}模型...")
    classifier.fit(X_balanced, y_balanced)
    
    # 创建完整的Pipeline (用于预测)
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    return pipeline, processed_messages

def evaluate_model(pipeline, val_messages, val_labels, output_dir, model_type, balance_method):
    """评估模型性能"""
    # 预处理验证集文本
    processed_val_messages = preprocess_text_for_training(val_messages)
    
    # 预测
    val_pred = pipeline.predict(processed_val_messages)
    
    # 获取决策分数
    if hasattr(pipeline.named_steps['classifier'], "decision_function"):
        # SVM有decision_function
        val_scores = pipeline.decision_function(processed_val_messages)
    elif hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        # NB和RF有predict_proba
        val_scores = pipeline.predict_proba(processed_val_messages)[:, 1]
    else:
        val_scores = None
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(val_labels, val_pred))
    
    # 如果有决策分数，绘制ROC曲线和PR曲线
    if val_scores is not None:
        # 创建结果目录
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
        
        # ROC曲线
        fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {model_type.upper()} Model ({balance_method})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "validation", f"{model_type}_{balance_method}_roc_curve.png"))
        
        # PR曲线
        precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall - {model_type.upper()} Model ({balance_method})')
        plt.savefig(os.path.join(output_dir, "validation", f"{model_type}_{balance_method}_pr_curve.png"))
        
        # 找到最佳决策阈值
        # 使用F1分数作为标准
        f1_scores = []
        for threshold in thresholds:
            y_pred = (val_scores >= threshold).astype(int)
            f1 = f1_score(val_labels, y_pred)
            f1_scores.append(f1)
        
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0
        best_f1 = f1_scores[best_threshold_idx] if best_threshold_idx < len(f1_scores) else 0
        
        print(f"最佳决策阈值: {best_threshold:.4f}, F1分数: {best_f1:.4f}")
        
        # 阈值-F1分数曲线
        plt.figure(figsize=(10, 8))
        # 修复：确保thresholds和f1_scores长度一致
        if len(thresholds) == len(f1_scores) - 1:
            plt.plot(thresholds, f1_scores[:-1], color='green', lw=2)  # precision_recall_curve返回的thresholds比precision和recall少一个
        else:
            # 如果长度不一致，使用相同长度的数组
            min_len = min(len(thresholds), len(f1_scores))
            plt.plot(thresholds[:min_len], f1_scores[:min_len], color='green', lw=2)
        plt.axvline(x=best_threshold, color='red', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'Threshold vs F1 - {model_type.upper()} Model ({balance_method})')
        plt.savefig(os.path.join(output_dir, "validation", f"{model_type}_{balance_method}_threshold_f1.png"))
        
        return best_threshold, best_f1
    
    return None, None

def export_model_params(pipeline, output_dir, processed_messages, model_type, balance_method, best_threshold=None):
    """导出模型参数为JSON格式"""
    # 获取向量化器
    vectorizer = pipeline.named_steps['vectorizer']
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    
    # 获取分类器
    classifier = pipeline.named_steps['classifier']
    
    # 创建基本模型参数字典
    model_params = {
        "vocabulary": vocabulary,
        "binary": vectorizer.binary,
        "lowercase": True,
        "otp_keywords": list(set(list(OTP_KEYWORDS['en'].keys()) + list(OTP_KEYWORDS['zh'].keys()) + 
                              list(OTP_KEYWORDS['ru'].keys()) + list(OTP_KEYWORDS['et'].keys()))),
        "model_type": model_type,
        "balance_method": balance_method
    }
    
    # 根据模型类型添加特定参数
    if model_type == 'svm':
        model_params["coef"] = classifier.coef_[0].tolist()
        model_params["intercept"] = float(classifier.intercept_[0])
    elif model_type == 'nb':
        model_params["feature_log_prob"] = classifier.feature_log_prob_.tolist()
        model_params["class_log_prior"] = classifier.class_log_prior_.tolist()
    elif model_type == 'rf':
        # 随机森林不容易导出为简单参数，这里我们只保存完整模型
        pass
    
    # 添加最佳决策阈值
    if best_threshold is not None:
        model_params["decision_threshold"] = float(best_threshold)
    
    # 保存为JSON
    params_path = os.path.join(output_dir, f"otp_{model_type}_{balance_method}_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(model_params, f, indent=2, cls=NumpyEncoder)
    
    print(f"模型参数已保存到: {params_path}")
    
    # 保存完整模型
    model_path = os.path.join(output_dir, f"otp_{model_type}_{balance_method}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"完整模型已保存到: {model_path}")
    
    # 保存一些示例处理后的文本，用于调试
    examples_path = os.path.join(output_dir, f"processed_examples_{model_type}_{balance_method}.txt")
    with open(examples_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(processed_messages[:20]):
            f.write(f"示例 {i+1}:\n{text}\n\n")
    
    print(f"处理后的示例文本已保存到: {examples_path}")
    
    return params_path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='结合增强特征工程和数据平衡策略训练OTP检测模型')
    parser.add_argument('--model-types', type=str, default='svm,nb,rf', help='模型类型，用逗号分隔 (默认: svm,nb,rf)')
    parser.add_argument('--balance-methods', type=str, default='smote,undersample', help='平衡方法，用逗号分隔 (默认: smote,undersample)')
    args = parser.parse_args()
    return args

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 解析模型类型和平衡方法
    model_types = args.model_types.split(',')
    balance_methods = args.balance_methods.split(',')
    
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 数据路径 - 使用combined_sms_dataset.csv数据集
    csv_file = os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    
    # 输出路径
    output_dir = os.path.join(base_dir, "models", "go_params")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    train_messages, train_labels, val_messages, val_labels = load_data_from_csv(csv_file)
    
    # 存储结果
    results = []
    
    # 对每种模型类型和平衡方法的组合进行训练
    for model_type in model_types:
        for balance_method in balance_methods:
            print(f"\n===== 训练{model_type.upper()}模型 (增强特征 + {balance_method}平衡) =====")
            
            # 训练模型
            pipeline, processed_messages = train_model(train_messages, train_labels, model_type, balance_method)
            
            # 评估模型
            best_threshold, best_f1 = evaluate_model(pipeline, val_messages, val_labels, output_dir, model_type, balance_method)
            
            # 导出模型参数
            export_model_params(pipeline, output_dir, processed_messages, model_type, balance_method, best_threshold)
            
            # 记录结果
            results.append({
                'model_type': model_type,
                'balance_method': balance_method,
                'best_threshold': best_threshold,
                'best_f1': best_f1
            })
    
    # 打印结果摘要
    print("\n===== 模型性能摘要 =====")
    for result in results:
        print(f"模型: {result['model_type'].upper()}, 平衡方法: {result['balance_method']}, "
              f"最佳阈值: {result['best_threshold']:.4f}, 最佳F1: {result['best_f1']:.4f}")
    
    print("\n所有模型训练完成!")

if __name__ == "__main__":
    main() 