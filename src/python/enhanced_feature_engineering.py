#!/usr/bin/env python3
"""
增强的特征工程模块 - 用于OTP短信检测
"""

import re
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import langdetect

# 增强的OTP关键词列表
OTP_KEYWORDS = {
    # 英文关键词及其权重
    "en": {
        "verification": 2.0,
        "code": 1.8,
        "otp": 2.5,
        "one-time": 2.2,
        "password": 1.5,
        "authenticate": 2.0,
        "secure": 1.3,
        "confirm": 1.2,
        "login": 1.5,
        "access": 1.2,
        "valid": 1.3,
        "expires": 1.4,
        "minutes": 1.1,
        "security": 1.4,
        "pin": 1.8,
        "token": 1.7,
        "authorization": 1.5,
        "verify": 2.0,
    },
    
    # 中文关键词及其权重
    "zh": {
        "验证码": 2.5,
        "验证": 1.8,
        "密码": 1.5,
        "一次性": 2.0,
        "有效期": 1.4,
        "失效": 1.3,
        "分钟": 1.1,
        "登录": 1.5,
        "登陆": 1.5,
        "安全": 1.2,
        "校验": 1.7,
        "确认": 1.3,
        "短信": 1.1,
        "动态码": 2.0,
        "授权码": 1.8,
        "识别码": 1.7,
        "认证码": 2.0,
        "临时": 1.4,
        "勿泄露": 1.6,
        "请勿": 1.2,
    },
    
    # 俄语关键词及其权重
    "ru": {
        "код": 2.5,           # 验证码
        "подтверждения": 2.0,  # 确认
        "пароль": 1.8,        # 密码
        "одноразовый": 2.2,    # 一次性
        "проверки": 1.8,       # 验证
        "авторизации": 1.7,    # 授权
        "действителен": 1.4,   # 有效
        "минут": 1.1,          # 分钟
        "безопасности": 1.3,   # 安全
        "вход": 1.5,           # 登录
        "доступ": 1.4,         # 访问
        "секретный": 1.6,      # 秘密
        "временный": 1.5,      # 临时
        "аутентификации": 1.9, # 认证
        "срок": 1.3,           # 期限
        "истекает": 1.4,       # 过期
    },
    
    # 爱沙尼亚语关键词及其权重
    "et": {
        "kood": 2.5,           # 验证码
        "kinnituskood": 2.2,   # 确认码
        "parool": 1.8,         # 密码
        "ühekordne": 2.0,      # 一次性
        "turva": 1.5,          # 安全
        "kehtib": 1.4,         # 有效
        "minutit": 1.1,        # 分钟
        "sisselogimine": 1.7,  # 登录
        "juurdepääs": 1.5,     # 访问
        "kinnitus": 1.8,       # 确认
        "autentimine": 2.0,    # 认证
        "ajutine": 1.4,        # 临时
        "aegub": 1.3,          # 过期
        "turvakood": 1.9,      # 安全码
        "salajane": 1.6,       # 秘密
        "tõendamine": 1.7,     # 验证
    }
}

# 数字模式特征
DIGIT_PATTERNS = {
    "has_4_digits": r"\b\d{4}\b",
    "has_5_digits": r"\b\d{5}\b",
    "has_6_digits": r"\b\d{6}\b",
    "has_8_digits": r"\b\d{8}\b",
    "has_consecutive_digits": r"\d{2,}",
    "has_digits_with_separator": r"\d+[\s\-\.]+\d+",
    "has_digits_in_brackets": r"[\[\(（【]\s*\d+\s*[\]\)）】]",
    "has_digits_after_colon": r"[:：]\s*\d+",
    "has_digits_with_prefix": r"[a-zA-Z]+\d+",
    "digits_percentage": None,  # 特殊处理，计算数字在文本中的比例
}

def detect_language(text):
    """
    检测文本语言
    返回: 'en' 英文, 'zh' 中文, 'ru' 俄语, 'et' 爱沙尼亚语, 'other' 其他语言
    """
    try:
        lang = langdetect.detect(text)
        if lang == 'zh-cn' or lang == 'zh-tw':
            return 'zh'
        elif lang == 'en':
            return 'en'
        elif lang == 'ru':
            return 'ru'
        elif lang == 'et':
            return 'et'
        else:
            return 'other'
    except:
        # 如果检测失败，尝试基于字符判断
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        russian_chars = len(re.findall(r'[а-яА-Я]', text))
        estonian_chars = len(re.findall(r'[õäöüÕÄÖÜ]', text))
        
        if chinese_chars > len(text) * 0.3:
            return 'zh'
        elif russian_chars > len(text) * 0.3:
            return 'ru'
        elif estonian_chars > len(text) * 0.1:
            return 'et'
        else:
            return 'en'  # 默认英文

def tokenize_text(text, language=None):
    """
    根据语言对文本进行分词
    """
    if language is None:
        language = detect_language(text)
    
    if language == 'zh':
        # 使用jieba分词处理中文
        return list(jieba.cut(text))
    else:
        # 英文、俄语、爱沙尼亚语和其他语言使用空格分词
        return text.split()

def extract_enhanced_features(text):
    """
    提取增强的特征
    """
    # 检测语言
    language = detect_language(text)
    
    # 基本清洗
    cleaned_text = clean_text(text)
    
    # 提取数字模式特征
    digit_features = extract_digit_patterns(text)
    
    # 提取关键词特征
    keyword_features = extract_keyword_features(text, language)
    
    # 分词（用于后续n-gram特征）
    tokens = tokenize_text(cleaned_text, language)
    
    # 构建特征字符串
    feature_str = cleaned_text
    
    # 添加数字模式特征
    for pattern, value in digit_features.items():
        if isinstance(value, bool) and value:
            feature_str += f" DIGIT_PATTERN_{pattern}"
        elif isinstance(value, float):
            # 对于百分比特征，将其离散化
            if pattern == "digits_percentage":
                if value > 0.3:
                    feature_str += " DIGIT_PATTERN_high_percentage"
                elif value > 0.1:
                    feature_str += " DIGIT_PATTERN_medium_percentage"
                elif value > 0:
                    feature_str += " DIGIT_PATTERN_low_percentage"
    
    # 添加关键词特征
    for keyword, weight in keyword_features.items():
        # 根据权重添加多次关键词特征，增强其影响
        repeat = int(weight * 2)  # 权重转换为重复次数
        for _ in range(repeat):
            feature_str += f" KEYWORD_{keyword}"
    
    # 添加语言标识特征
    feature_str += f" LANG_{language}"
    
    return feature_str, tokens, language

def clean_text(text):
    """
    清洗文本
    """
    # 转为小写
    text = text.lower()
    
    # 替换标点符号为空格
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    # 标准化空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def extract_digit_patterns(text):
    """
    提取数字模式特征
    """
    features = {}
    
    # 提取各种数字模式
    for pattern_name, pattern in DIGIT_PATTERNS.items():
        if pattern_name == "digits_percentage":
            # 计算数字在文本中的比例
            digits = re.findall(r'\d', text)
            features[pattern_name] = len(digits) / len(text) if len(text) > 0 else 0
        else:
            # 检查是否匹配模式
            matches = re.findall(pattern, text)
            features[pattern_name] = len(matches) > 0
    
    return features

def extract_keyword_features(text, language):
    """
    提取关键词特征
    """
    text_lower = text.lower()
    features = {}
    
    # 获取对应语言的关键词列表
    keywords = OTP_KEYWORDS.get(language, OTP_KEYWORDS['en'])
    
    # 检查每个关键词
    for keyword, weight in keywords.items():
        if keyword in text_lower:
            features[keyword] = weight
    
    # 对于其他语言，也检查英文关键词
    if language not in ['en']:
        for keyword, weight in OTP_KEYWORDS['en'].items():
            if keyword in text_lower:
                features[keyword] = weight
    
    return features

def create_enhanced_vectorizer(max_features=3000, ngram_range=(1, 3)):
    """
    创建增强的特征向量化器
    """
    return CountVectorizer(
        lowercase=True,
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        ngram_range=ngram_range,
        binary=True
    )

def preprocess_text_for_training(messages):
    """
    为训练准备文本数据
    """
    processed_messages = []
    
    for message in messages:
        # 提取增强特征
        feature_str, _, _ = extract_enhanced_features(message)
        processed_messages.append(feature_str)
    
    return processed_messages 