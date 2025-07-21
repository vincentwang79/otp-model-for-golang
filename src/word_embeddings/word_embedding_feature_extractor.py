#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于词嵌入的特征提取器
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import logging
import jieba
import string

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordEmbeddingFeatureExtractor:
    """
    基于词嵌入的特征提取器
    用于从文本中提取词嵌入特征，并与其他特征融合
    """
    
    def __init__(self, embedding_model=None, embedding_dim=300, language='auto'):
        """
        初始化特征提取器
        
        Args:
            embedding_model: 词嵌入模型
            embedding_dim: 嵌入维度
            language: 语言，可选值：'auto', 'en', 'zh', 'ru', 'et'
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.language = language
        
        # OTP相关关键词
        self.otp_keywords = {
            'en': ['code', 'verification', 'verify', 'otp', 'password', 'pin', 'authenticate', 'login'],
            'zh': ['验证码', '校验码', '确认码', '动态密码', '验证', '登录', '登陆', '密码'],
            'ru': ['код', 'подтверждение', 'пароль', 'пин', 'авторизация', 'вход'],
            'et': ['kood', 'kinnituskood', 'parool', 'sisselogimine']
        }
    
    def detect_language(self, text):
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码: 'en', 'zh', 'ru', 'et'
        """
        # 简单的语言检测逻辑
        # 实际应用中应使用更复杂的语言检测库
        
        # 检测中文
        if any('\u4e00' <= ch <= '\u9fff' for ch in text):
            return 'zh'
        
        # 检测俄语 (西里尔字母)
        if any('\u0400' <= ch <= '\u04FF' for ch in text):
            return 'ru'
        
        # 检测爱沙尼亚语 (特殊字符)
        estonian_chars = set('õäöüÕÄÖÜ')
        if any(ch in estonian_chars for ch in text):
            return 'et'
        
        # 默认为英语
        return 'en'
    
    def preprocess_text(self, text, language=None):
        """
        预处理文本
        
        Args:
            text: 输入文本
            language: 语言代码
            
        Returns:
            分词后的文本列表
        """
        if language is None:
            language = self.language
            
        if language == 'auto':
            language = self.detect_language(text)
        
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 根据语言进行分词
        if language == 'zh':
            # 中文分词
            tokens = list(jieba.cut(text))
        else:
            # 其他语言简单按空格分词
            tokens = text.split()
        
        # 移除空白标记
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def extract_number_patterns(self, text):
        """
        提取数字模式特征
        
        Args:
            text: 输入文本
            
        Returns:
            数字模式特征字典
        """
        features = {}
        
        # 提取4-8位数字
        otp_patterns = re.findall(r'\b\d{4,8}\b', text)
        features['has_potential_otp'] = 1 if otp_patterns else 0
        features['otp_pattern_count'] = len(otp_patterns)
        
        # 数字密度
        digits = sum(c.isdigit() for c in text)
        features['digit_density'] = digits / max(1, len(text))
        
        # 数字模式的位置
        if otp_patterns:
            # 找出第一个OTP模式在文本中的相对位置
            first_pos = text.find(otp_patterns[0])
            features['otp_relative_position'] = first_pos / max(1, len(text))
        else:
            features['otp_relative_position'] = -1
            
        return features
    
    def extract_keyword_features(self, tokens, language):
        """
        提取关键词特征
        
        Args:
            tokens: 分词后的文本
            language: 语言代码
            
        Returns:
            关键词特征字典
        """
        features = {}
        
        # 获取对应语言的关键词列表
        keywords = self.otp_keywords.get(language, self.otp_keywords['en'])
        
        # 计算关键词出现次数
        keyword_count = sum(1 for token in tokens if token in keywords)
        features['otp_keyword_count'] = keyword_count
        features['has_otp_keyword'] = 1 if keyword_count > 0 else 0
        
        # 关键词密度
        features['keyword_density'] = keyword_count / max(1, len(tokens))
        
        return features
    
    def get_word_embedding(self, word):
        """
        获取词的嵌入向量
        
        Args:
            word: 输入词
            
        Returns:
            词嵌入向量
        """
        # 如果有预训练模型，应该使用模型获取词向量
        # 由于环境限制，这里返回随机向量
        return np.random.rand(self.embedding_dim)
    
    def get_text_embedding(self, tokens):
        """
        获取文本的嵌入向量
        
        Args:
            tokens: 分词后的文本
            
        Returns:
            文本嵌入向量
        """
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # 获取每个词的嵌入向量
        word_embeddings = [self.get_word_embedding(token) for token in tokens]
        
        # 计算平均词向量作为文本表示
        text_embedding = np.mean(word_embeddings, axis=0)
        
        return text_embedding
    
    def extract_features(self, text):
        """
        提取所有特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        # 检测语言
        language = self.detect_language(text) if self.language == 'auto' else self.language
        
        # 预处理文本
        tokens = self.preprocess_text(text, language)
        
        # 提取数字模式特征
        number_features = self.extract_number_patterns(text)
        
        # 提取关键词特征
        keyword_features = self.extract_keyword_features(tokens, language)
        
        # 提取词嵌入特征
        embedding = self.get_text_embedding(tokens)
        
        # 合并所有特征
        features = {**number_features, **keyword_features}
        
        # 将词嵌入特征添加到特征字典
        for i, value in enumerate(embedding):
            features[f'embedding_{i}'] = value
            
        return features
    
    def transform(self, texts):
        """
        转换文本列表为特征矩阵
        
        Args:
            texts: 文本列表
            
        Returns:
            特征矩阵
        """
        features_list = []
        
        for text in texts:
            features = self.extract_features(text)
            features_list.append(features)
        
        # 转换为DataFrame
        df_features = pd.DataFrame(features_list)
        
        return df_features

def main():
    """主函数"""
    # 示例文本
    texts = [
        "Your verification code is 123456",
        "您的验证码是987654，请勿泄露",
        "Ваш код подтверждения: 246810",
        "Teie kinnituskood on 135790"
    ]
    
    # 初始化特征提取器
    extractor = WordEmbeddingFeatureExtractor()
    
    # 提取特征
    features = extractor.transform(texts)
    
    # 打印特征
    print(features.head())
    
    # 打印特征维度
    print(f"特征维度: {features.shape}")

if __name__ == "__main__":
    main() 