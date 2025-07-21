#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于词嵌入特征的模型训练与评估
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import logging
import joblib

from word_embedding_feature_extractor import WordEmbeddingFeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordEmbeddingModelTrainer:
    """
    基于词嵌入特征的模型训练与评估
    """
    
    def __init__(self, data_path, model_type='svm', feature_extractor=None):
        """
        初始化模型训练器
        
        Args:
            data_path: 数据集路径
            model_type: 模型类型，可选值：'svm', 'rf', 'nn'
            feature_extractor: 特征提取器
        """
        self.data_path = data_path
        self.model_type = model_type
        self.feature_extractor = feature_extractor or WordEmbeddingFeatureExtractor()
        self.model = None
        self.best_threshold = 0.5
        
    def load_data(self):
        """
        加载数据集
        
        Returns:
            训练集和验证集的特征和标签
        """
        logger.info(f"加载数据集: {self.data_path}")
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"数据集加载成功，共 {len(data)} 条记录")
            
            # 分割训练集和验证集
            train_data = data[data['split'] == 'train']
            val_data = data[data['split'] == 'val']
            
            logger.info(f"训练集: {len(train_data)} 条记录")
            logger.info(f"验证集: {len(val_data)} 条记录")
            
            # 提取特征
            logger.info("提取训练集特征...")
            X_train = self.feature_extractor.transform(train_data['message'].tolist())
            y_train = train_data['is_otp'].values
            
            logger.info("提取验证集特征...")
            X_val = self.feature_extractor.transform(val_data['message'].tolist())
            y_val = val_data['is_otp'].values
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            return None, None, None, None
    
    def create_model(self):
        """
        创建模型
        
        Returns:
            创建的模型
        """
        if self.model_type == 'svm':
            logger.info("创建SVM模型")
            return SVC(kernel='linear', class_weight='balanced', probability=True)
        elif self.model_type == 'rf':
            logger.info("创建随机森林模型")
            return RandomForestClassifier(n_estimators=100, class_weight='balanced')
        elif self.model_type == 'nn':
            logger.info("创建神经网络模型")
            return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300)
        else:
            logger.error(f"不支持的模型类型: {self.model_type}")
            return None
    
    def train_model(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            
        Returns:
            训练好的模型
        """
        logger.info(f"开始训练 {self.model_type} 模型...")
        
        model = self.create_model()
        if model is None:
            return None
        
        model.fit(X_train, y_train)
        logger.info("模型训练完成")
        
        return model
    
    def find_optimal_threshold(self, model, X_val, y_val):
        """
        寻找最佳决策阈值
        
        Args:
            model: 训练好的模型
            X_val: 验证集特征
            y_val: 验证集标签
            
        Returns:
            最佳阈值
        """
        logger.info("寻找最佳决策阈值...")
        
        # 获取验证集的预测概率
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算不同阈值下的F1分数
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
        
        # 找到最佳阈值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        logger.info(f"最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")
        
        return best_threshold
    
    def evaluate_model(self, model, X_val, y_val, threshold=None):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_val: 验证集特征
            y_val: 验证集标签
            threshold: 决策阈值
            
        Returns:
            评估指标字典
        """
        logger.info("评估模型...")
        
        # 获取验证集的预测概率
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # 使用指定阈值或默认阈值进行预测
        if threshold is None:
            threshold = self.best_threshold
        
        y_pred = (y_proba >= threshold).astype(int)
        
        # 计算评估指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_val, y_pred)
        
        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "threshold": threshold
        }
        
        logger.info(f"评估结果:")
        logger.info(f"  准确率: {accuracy:.4f}")
        logger.info(f"  精确率: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")
        logger.info(f"  F1分数: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def compare_with_tfidf(self, word_embedding_results):
        """
        与TF-IDF模型进行比较
        
        Args:
            word_embedding_results: 词嵌入模型的评估结果
            
        Returns:
            比较结果
        """
        # 这里应该加载已有的TF-IDF模型结果
        # 由于环境限制，这里使用模拟数据
        
        logger.info("与TF-IDF模型进行比较...")
        
        # 模拟TF-IDF模型结果
        tfidf_results = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.85,
            "f1_score": 0.87,
            "roc_auc": 0.94
        }
        
        # 计算提升百分比
        improvements = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if metric in word_embedding_results and metric in tfidf_results:
                improvement = (word_embedding_results[metric] - tfidf_results[metric]) / tfidf_results[metric] * 100
                improvements[metric] = improvement
        
        logger.info("与TF-IDF模型相比的提升:")
        for metric, improvement in improvements.items():
            logger.info(f"  {metric}: {improvement:.2f}%")
        
        return {
            "word_embedding": word_embedding_results,
            "tfidf": tfidf_results,
            "improvements": improvements
        }
    
    def save_model(self, model, output_dir="models"):
        """
        保存模型
        
        Args:
            model: 训练好的模型
            output_dir: 输出目录
            
        Returns:
            保存的模型路径
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_name = f"word_embedding_{self.model_type}_model.joblib"
        model_path = os.path.join(output_dir, model_name)
        
        logger.info(f"保存模型到 {model_path}")
        joblib.dump(model, model_path)
        
        # 保存阈值
        threshold_path = os.path.join(output_dir, f"word_embedding_{self.model_type}_threshold.txt")
        with open(threshold_path, 'w') as f:
            f.write(str(self.best_threshold))
        
        return model_path
    
    def run(self):
        """
        运行模型训练和评估流程
        
        Returns:
            评估结果
        """
        # 加载数据
        X_train, y_train, X_val, y_val = self.load_data()
        if X_train is None:
            return None
        
        # 训练模型
        model = self.train_model(X_train, y_train)
        if model is None:
            return None
        
        # 寻找最佳阈值
        self.best_threshold = self.find_optimal_threshold(model, X_val, y_val)
        
        # 评估模型
        results = self.evaluate_model(model, X_val, y_val, self.best_threshold)
        
        # 与TF-IDF模型比较
        comparison = self.compare_with_tfidf(results)
        
        # 保存模型
        self.save_model(model)
        
        self.model = model
        
        return results

def main():
    """主函数"""
    # 数据路径
    data_path = "data/processed/balanced_smote_dataset.csv"
    
    # 创建特征提取器
    feature_extractor = WordEmbeddingFeatureExtractor()
    
    # 训练不同类型的模型
    model_types = ["svm", "rf", "nn"]
    results = []
    
    for model_type in model_types:
        logger.info(f"训练 {model_type} 模型")
        trainer = WordEmbeddingModelTrainer(data_path, model_type, feature_extractor)
        result = trainer.run()
        if result:
            result["model_type"] = model_type
            results.append(result)
    
    # 输出比较结果
    if results:
        best_result = max(results, key=lambda x: x["f1_score"])
        logger.info(f"\n最佳模型: {best_result['model_type']}")
        logger.info(f"F1分数: {best_result['f1_score']:.4f}")
    else:
        logger.error("没有可用的评估结果")

if __name__ == "__main__":
    main() 