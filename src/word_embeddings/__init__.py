"""
词嵌入OTP检测器 - Python模块

本模块包含用于训练和评估基于词嵌入的OTP检测模型的工具。
"""

from .word_embeddings import WordEmbeddingEvaluator
from .word_embedding_feature_extractor import WordEmbeddingFeatureExtractor
from .word_embedding_model_trainer import WordEmbeddingModelTrainer

__all__ = [
    'WordEmbeddingEvaluator',
    'WordEmbeddingFeatureExtractor',
    'WordEmbeddingModelTrainer'
] 