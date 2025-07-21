#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
词嵌入模型评估和选择模块
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordEmbeddingEvaluator:
    """
    词嵌入模型评估器
    用于评估不同词嵌入模型在OTP检测任务上的性能
    """
    
    def __init__(self, data_path, model_type="fasttext"):
        """
        初始化评估器
        
        Args:
            data_path: 数据集路径
            model_type: 词嵌入模型类型，可选值：word2vec, glove, fasttext, bert
        """
        self.data_path = data_path
        self.model_type = model_type
        self.data = None
        self.model = None
        self.embedding_dim = 300  # 默认嵌入维度
        
    def load_data(self):
        """加载数据集"""
        logger.info(f"加载数据集: {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"数据集加载成功，共 {len(self.data)} 条记录")
            # 检查数据集是否包含必要的列
            required_cols = ['message', 'is_otp', 'split']
            for col in required_cols:
                if col not in self.data.columns:
                    logger.error(f"数据集缺少必要的列: {col}")
                    return False
            return True
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            return False
    
    def load_embedding_model(self):
        """
        加载词嵌入模型
        
        注意: 这里只是占位，实际实现需要根据环境中可用的库
        """
        logger.info(f"加载 {self.model_type} 词嵌入模型")
        
        # 这里应该根据self.model_type加载相应的预训练模型
        # 由于环境限制，这里只是模拟加载过程
        logger.info("注意: 当前只是模拟加载模型，未实际加载")
        
        # 模拟成功加载
        return True
    
    def extract_features(self, texts):
        """
        从文本中提取词嵌入特征
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy数组，每行是一个文本的词嵌入特征
        """
        # 这里应该使用加载的词嵌入模型提取特征
        # 由于环境限制，这里只是返回随机特征作为示例
        logger.info(f"为 {len(texts)} 条文本提取 {self.model_type} 特征")
        
        # 模拟特征提取，实际应该使用词嵌入模型
        features = np.random.rand(len(texts), self.embedding_dim)
        return features
    
    def evaluate_model(self):
        """
        评估词嵌入模型在OTP检测任务上的性能
        
        Returns:
            dict: 包含评估指标的字典
        """
        if not self.load_data():
            return None
        
        if not self.load_embedding_model():
            return None
        
        # 分割训练集和验证集
        train_data = self.data[self.data['split'] == 'train']
        val_data = self.data[self.data['split'] == 'val']
        
        # 提取特征
        X_train = self.extract_features(train_data['message'].tolist())
        y_train = train_data['is_otp'].values
        
        X_val = self.extract_features(val_data['message'].tolist())
        y_val = val_data['is_otp'].values
        
        # 训练简单的SVM分类器
        logger.info("训练SVM分类器")
        clf = SVC(kernel='linear', class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # 预测并评估
        y_pred = clf.predict(X_val)
        
        # 计算评估指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        results = {
            "model_type": self.model_type,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        logger.info(f"评估结果: {results}")
        return results

def main():
    """主函数"""
    # 数据路径
    data_path = "data/processed/balanced_smote_dataset.csv"
    
    # 评估不同的词嵌入模型
    model_types = ["word2vec", "glove", "fasttext", "bert"]
    results = []
    
    for model_type in model_types:
        logger.info(f"评估 {model_type} 模型")
        evaluator = WordEmbeddingEvaluator(data_path, model_type)
        result = evaluator.evaluate_model()
        if result:
            results.append(result)
    
    # 输出比较结果
    if results:
        df_results = pd.DataFrame(results)
        logger.info("\n词嵌入模型比较结果:")
        logger.info(f"\n{df_results}")
    else:
        logger.error("没有可用的评估结果")

if __name__ == "__main__":
    main() 