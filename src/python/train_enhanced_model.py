#!/usr/bin/env python3
"""
使用增强特征工程训练OTP检测模型
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import joblib
import json
import matplotlib.pyplot as plt
import time

# 导入增强特征工程模块
from enhanced_feature_engineering import preprocess_text_for_training, create_enhanced_vectorizer

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
    
    return train_messages, train_labels, val_messages, val_labels

def train_model(messages, labels, model_type='svm'):
    """训练OTP检测模型"""
    start_time = time.time()
    
    # 预处理文本，使用增强特征
    print("预处理文本，提取增强特征...")
    processed_messages = preprocess_text_for_training(messages)
    
    # 创建增强的向量化器
    vectorizer = create_enhanced_vectorizer(max_features=3000, ngram_range=(1, 3))
    
    # 根据模型类型创建分类器
    if model_type == 'svm':
        # 线性SVM分类器
        classifier = LinearSVC(
            C=1.0,
            class_weight={0: 1, 1: 5},  # 增加OTP类别的权重
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
            class_weight={0: 1, 1: 5}
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建Pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])
    
    # 训练模型
    print(f"训练{model_type.upper()}模型...")
    pipeline.fit(processed_messages, labels)
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    return pipeline, processed_messages

def evaluate_model(pipeline, val_messages, val_labels, output_dir, model_type):
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
        plt.title(f'Receiver Operating Characteristic - {model_type.upper()} Model')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "validation", f"{model_type}_roc_curve.png"))
        
        # PR曲线
        precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_type.upper()} Model')
        plt.savefig(os.path.join(output_dir, "validation", f"{model_type}_pr_curve.png"))
        
        # 找到最佳决策阈值
        # 使用F1分数作为标准
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0
        
        print(f"最佳决策阈值: {best_threshold:.4f}")
        
        return best_threshold
    
    return None

def export_model_params(pipeline, output_dir, processed_messages, model_type, best_threshold=None, balance_method='enhanced'):
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
        "otp_keywords": list(set(list(OTP_KEYWORDS['en'].keys()) + list(OTP_KEYWORDS['zh'].keys()))),
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

def main():
    # 导入增强特征工程中的关键词列表
    global OTP_KEYWORDS
    from enhanced_feature_engineering import OTP_KEYWORDS
    
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
    
    # 训练不同类型的模型
    model_types = ['svm', 'nb', 'rf']
    balance_method = 'enhanced'  # 使用增强特征工程
    
    for model_type in model_types:
        print(f"\n===== 训练{model_type.upper()}模型 (增强特征) =====")
        
        # 训练模型
        pipeline, processed_messages = train_model(train_messages, train_labels, model_type)
        
        # 评估模型
        best_threshold = evaluate_model(pipeline, val_messages, val_labels, output_dir, model_type)
        
        # 导出模型参数
        export_model_params(pipeline, output_dir, processed_messages, model_type, best_threshold, balance_method)
    
    print("\n所有模型训练完成!")

if __name__ == "__main__":
    main() 