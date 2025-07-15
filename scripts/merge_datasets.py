 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新的SMS数据集整合到现有数据集中
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_datasets():
    """加载现有数据集和新数据集"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 数据文件路径
    existing_dataset_path = os.path.join(base_dir, "data", "processed", "combined_sms_dataset.csv")
    new_dataset_path = os.path.join(base_dir, "data", "raw", "sms_results_base.csv")
    
    # 加载现有数据集
    print("加载现有数据集: {}".format(existing_dataset_path))
    existing_df = pd.read_csv(existing_dataset_path)
    
    # 加载新数据集
    print("加载新数据集: {}".format(new_dataset_path))
    new_df = pd.read_csv(new_dataset_path)
    
    # 打印数据集信息
    print("现有数据集大小: {}".format(len(existing_df)))
    print("新数据集大小: {}".format(len(new_df)))
    
    return existing_df, new_df

def preprocess_new_dataset(new_df):
    """预处理新数据集，使其与现有数据集格式一致"""
    # 只保留需要的列
    if 'message' in new_df.columns and 'is_otp' in new_df.columns:
        # 选择需要的列
        new_df = new_df[['message', 'is_otp']]
        
        # 将is_otp列从布尔值转换为整数 (True->1, False->0)
        new_df['is_otp'] = new_df['is_otp'].astype(int)
        
        # 移除重复消息
        new_df = new_df.drop_duplicates(subset=['message'])
        
        # 移除空消息
        new_df = new_df.dropna(subset=['message'])
        
        print("预处理后新数据集大小: {}".format(len(new_df)))
        return new_df
    else:
        raise ValueError("新数据集缺少必要的列 'message' 或 'is_otp'")

def split_new_dataset(new_df):
    """将新数据集分为训练集和验证集"""
    # 按照与现有数据集相同的比例分割 (90% 训练, 10% 验证)
    train_df, val_df = train_test_split(
        new_df, 
        test_size=0.1, 
        random_state=42, 
        stratify=new_df['is_otp']
    )
    
    # 添加split列
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # 合并回一个数据框
    new_df_with_split = pd.concat([train_df, val_df])
    
    # 打印分割信息
    print("新数据训练集大小: {}".format(len(train_df)))
    print("新数据验证集大小: {}".format(len(val_df)))
    
    return new_df_with_split

def merge_datasets(existing_df, new_df_with_split):
    """合并现有数据集和新数据集"""
    # 合并数据集
    merged_df = pd.concat([existing_df, new_df_with_split], ignore_index=True)
    
    # 移除重复消息，保留第一次出现的
    merged_df = merged_df.drop_duplicates(subset=['message'], keep='first')
    
    # 打印合并后的数据集信息
    print("合并后数据集大小: {}".format(len(merged_df)))
    print("OTP消息数量: {}".format(merged_df['is_otp'].sum()))
    print("非OTP消息数量: {}".format(len(merged_df) - merged_df['is_otp'].sum()))
    print("训练集大小: {}".format(len(merged_df[merged_df['split'] == 'train'])))
    print("验证集大小: {}".format(len(merged_df[merged_df['split'] == 'val'])))
    
    return merged_df

def generate_stats(merged_df):
    """生成数据集统计信息"""
    total = len(merged_df)
    otp_count = merged_df['is_otp'].sum()
    non_otp_count = total - otp_count
    
    train_df = merged_df[merged_df['split'] == 'train']
    val_df = merged_df[merged_df['split'] == 'val']
    
    train_count = len(train_df)
    val_count = len(val_df)
    
    train_otp_count = train_df['is_otp'].sum()
    train_non_otp_count = train_count - train_otp_count
    
    val_otp_count = val_df['is_otp'].sum()
    val_non_otp_count = val_count - val_otp_count
    
    stats = """# 数据集统计信息

总样本数: {total}
OTP短信: {otp_count} ({otp_percent:.1f}%)
非OTP短信: {non_otp_count} ({non_otp_percent:.1f}%)

## 训练集/验证集划分

训练集: {train_count} ({train_percent:.1f}%)
  - OTP短信: {train_otp_count}
  - 非OTP短信: {train_non_otp_count}

验证集: {val_count} ({val_percent:.1f}%)
  - OTP短信: {val_otp_count}
  - 非OTP短信: {val_non_otp_count}
""".format(
        total=total,
        otp_count=otp_count,
        otp_percent=otp_count/total*100,
        non_otp_count=non_otp_count,
        non_otp_percent=non_otp_count/total*100,
        train_count=train_count,
        train_percent=train_count/total*100,
        train_otp_count=train_otp_count,
        train_non_otp_count=train_non_otp_count,
        val_count=val_count,
        val_percent=val_count/total*100,
        val_otp_count=val_otp_count,
        val_non_otp_count=val_non_otp_count
    )
    return stats

def save_results(merged_df, stats):
    """保存合并后的数据集和统计信息"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 输出文件路径
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并后的数据集
    output_path = os.path.join(output_dir, "combined_sms_dataset_updated.csv")
    merged_df.to_csv(output_path, index=False)
    print("合并后的数据集已保存到: {}".format(output_path))
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, "dataset_stats_updated.md")
    with open(stats_path, 'w') as f:
        f.write(stats)
    print("统计信息已保存到: {}".format(stats_path))
    
    # 创建备份
    backup_path = os.path.join(output_dir, "combined_sms_dataset_backup.csv")
    original_path = os.path.join(output_dir, "combined_sms_dataset.csv")
    
    # 复制原始数据集作为备份
    original_df = pd.read_csv(original_path)
    original_df.to_csv(backup_path, index=False)
    print("原始数据集已备份到: {}".format(backup_path))

def main():
    """主函数"""
    try:
        # 加载数据集
        existing_df, new_df = load_datasets()
        
        # 预处理新数据集
        new_df = preprocess_new_dataset(new_df)
        
        # 分割新数据集
        new_df_with_split = split_new_dataset(new_df)
        
        # 合并数据集
        merged_df = merge_datasets(existing_df, new_df_with_split)
        
        # 生成统计信息
        stats = generate_stats(merged_df)
        
        # 保存结果
        save_results(merged_df, stats)
        
        print("数据集合并完成!")
        
    except Exception as e:
        print("错误: {}".format(e))

if __name__ == "__main__":
    main() 