#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统 - 数据集检查工具
用于验证数据转换结果是否正确
"""

import os
import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import logging
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_class_distribution(data_root, output_dir):
    """
    可视化各类别的样本分布
    
    Args:
        data_root: 数据集根目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计各个类别的样本数量
    splits = ['train', 'val', 'test']
    class_counts = {split: {} for split in splits}
    
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            logger.warning(f"目录不存在: {split_dir}")
            continue
            
        # 获取所有类别
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        # 统计每个类别的样本数量
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[split][cls] = len(files)
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    
    for i, split in enumerate(splits):
        if not class_counts[split]:
            continue
            
        classes = list(class_counts[split].keys())
        counts = list(class_counts[split].values())
        
        plt.subplot(1, len(splits), i+1)
        plt.bar(classes, counts)
        plt.title(f'{split.capitalize()} Set')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # 在柱子上显示数量
        for j, count in enumerate(counts):
            plt.text(j, count + 1, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    logger.info(f"类别分布图已保存: {os.path.join(output_dir, 'class_distribution.png')}")

def visualize_sample_images(data_root, output_dir, samples_per_class=2):
    """
    可视化每个类别的样本图像
    
    Args:
        data_root: 数据集根目录
        output_dir: 输出目录
        samples_per_class: 每个类别展示的样本数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有类别
    train_dir = os.path.join(data_root, 'train')
    if not os.path.exists(train_dir):
        logger.warning(f"训练集目录不存在: {train_dir}")
        return
        
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # 对于每个类别，随机选择样本并显示
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not files:
            logger.warning(f"类别 {cls} 没有样本")
            continue
        
        # 随机选择样本
        selected_files = random.sample(files, min(samples_per_class, len(files)))
        
        # 创建类别目录
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        # 复制样本图像
        for file in selected_files:
            src_path = os.path.join(cls_dir, file)
            dst_path = os.path.join(output_dir, cls, file)
            
            # 读取图像
            img = cv2.imread(src_path)
            # 添加文字
            cv2.putText(img, f"Class: {cls}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 保存图像
            cv2.imwrite(dst_path, img)
            
            logger.info(f"已保存样本: {dst_path}")

def check_yolo_to_cnn_conversion(yolo_data_yaml, cnn_data_root, output_dir):
    """
    检查YOLO到CNN的转换结果
    
    Args:
        yolo_data_yaml: YOLO数据集的yaml文件路径
        cnn_data_root: CNN数据集根目录
        output_dir: 输出目录
    """
    # 读取YOLO配置
    with open(yolo_data_yaml, 'r') as f:
        yolo_config = yaml.safe_load(f)
    
    class_names = yolo_config['names']
    logger.info(f"YOLO类别名称: {class_names}")
    
    # 检查CNN数据集类别
    train_dir = os.path.join(cnn_data_root, 'train')
    if not os.path.exists(train_dir):
        logger.error(f"CNN训练集目录不存在: {train_dir}")
        return
    
    cnn_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    logger.info(f"CNN类别名称: {cnn_classes}")
    
    # 检查类别是否匹配
    if set(class_names) == set(cnn_classes):
        logger.info("✓ YOLO和CNN类别名称完全匹配")
    else:
        logger.warning("⚠ YOLO和CNN类别名称不匹配")
        # 找出不匹配的类别
        missing_in_cnn = set(class_names) - set(cnn_classes)
        if missing_in_cnn:
            logger.warning(f"  YOLO中有但CNN中没有的类别: {missing_in_cnn}")
        
        missing_in_yolo = set(cnn_classes) - set(class_names)
        if missing_in_yolo:
            logger.warning(f"  CNN中有但YOLO中没有的类别: {missing_in_yolo}")
    
    # 创建报告文件
    report_path = os.path.join(output_dir, 'conversion_report.txt')
    with open(report_path, 'w') as f:
        f.write("YOLO到CNN转换检查报告\n")
        f.write("======================\n\n")
        
        f.write(f"YOLO类别名称: {class_names}\n")
        f.write(f"CNN类别名称: {cnn_classes}\n\n")
        
        # 统计各类别样本数量
        f.write("CNN数据集统计\n")
        f.write("--------------\n")
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(cnn_data_root, split)
            if not os.path.exists(split_dir):
                f.write(f"{split}集目录不存在\n")
                continue
                
            f.write(f"\n{split.capitalize()}集:\n")
            
            for cls in cnn_classes:
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.exists(cls_dir):
                    f.write(f"  {cls}: 目录不存在\n")
                    continue
                    
                files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                f.write(f"  {cls}: {len(files)}个样本\n")
    
    logger.info(f"转换检查报告已保存: {report_path}")

def main():
    """主函数"""
    # 设置路径
    cnn_data_root = 'data/welding_defects/cnn'
    yolo_data_yaml = 'data/welding_defects/yolov8/data.yaml'
    output_dir = 'results/dataset_check'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("开始检查数据集")
    
    # 检查YOLO到CNN的转换
    check_yolo_to_cnn_conversion(yolo_data_yaml, cnn_data_root, output_dir)
    
    # 可视化类别分布
    logger.info("可视化类别分布")
    visualize_class_distribution(cnn_data_root, output_dir)
    
    # 可视化样本图像
    logger.info("可视化样本图像")
    visualize_sample_images(cnn_data_root, os.path.join(output_dir, 'samples'))
    
    logger.info(f"数据集检查完成，结果保存在: {output_dir}")
    
    # 打印一些建议
    print("\n========== 数据集检查结果 ==========")
    print(f"详细报告已保存在: {output_dir}")
    print("\n建议:")
    print("1. 检查类别分布是否均衡，如果某些类别样本太少可能需要数据增强")
    print("2. 检查样本图像是否符合预期，特别是裁剪区域是否正确包含焊缝缺陷")
    print("3. 如果数据集分布良好，可以继续进行模型训练")
    print("======================================")

if __name__ == '__main__':
    main() 