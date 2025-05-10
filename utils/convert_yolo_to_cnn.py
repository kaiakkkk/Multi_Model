#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将YOLOv8数据集转换为CNN训练所需格式的工具脚本

YOLOv8格式: 
  - data.yaml 包含类别信息
  - images/xxx.jpg 图像文件
  - labels/xxx.txt 标签文件，格式为: <class_id> <center_x> <center_y> <width> <height>

CNN格式:
  - data_root/{train,val,test}/{类别名称}/xxx.jpg
"""

import os
import cv2
import yaml
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_yaml_config(yaml_path):
    """读取YAML配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_min_enclosing_rect(points):
    """获取多边形的最小包围矩形
    
    Args:
        points: 多边形顶点坐标，形状为 [N, 2]
    
    Returns:
        [x_min, y_min, x_max, y_max]: 矩形框坐标 (归一化)
    """
    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    return [x_min, y_min, x_max, y_max]

def parse_yolo_label(label_path, img_width, img_height):
    """解析YOLO标签文件，将多边形转换为矩形框
    
    Args:
        label_path: 标签文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        list of [class_id, x_min, y_min, x_max, y_max]
    """
    if not os.path.exists(label_path):
        logger.warning(f"标签文件不存在: {label_path}")
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            # 将所有坐标点转换为点列表
            polygon_points = []
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    x = float(parts[i])
                    y = float(parts[i+1])
                    polygon_points.append([x, y])
            
            if not polygon_points:
                continue
                
            # 获取多边形的最小包围矩形
            x_min, y_min, x_max, y_max = get_min_enclosing_rect(polygon_points)
            
            # 确保边界在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(1, x_max)
            y_max = min(1, y_max)
            
            # 只有当边界框有效时才添加
            if x_min < x_max and y_min < y_max:
                boxes.append([class_id, x_min, y_min, x_max, y_max])
            else:
                logger.warning(f"无效边界框: {[x_min, y_min, x_max, y_max]} in {label_path}")
                
    return boxes

def split_dataset(files, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """将数据集分割为训练集、验证集和测试集
    
    Args:
        files: 文件路径列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_files, val_files, test_files: 分割后的文件列表
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"
    
    random.seed(seed)
    random.shuffle(files)
    
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files

def create_cnn_dataset(yolo_dataset_path, output_path, class_names, split_data=True):
    """创建CNN数据集
    
    Args:
        yolo_dataset_path: YOLO数据集路径
        output_path: 输出路径
        class_names: 类别名称
        split_data: 是否分割数据集
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建类别目录
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_path, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # 获取训练集图像列表
    train_img_dir = os.path.join(yolo_dataset_path, 'train', 'images')
    train_label_dir = os.path.join(yolo_dataset_path, 'train', 'labels')
    
    if not os.path.exists(train_img_dir):
        logger.error(f"训练集图像目录不存在: {train_img_dir}")
        return
    
    image_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 分割数据集
    if split_data:
        train_images, val_images, test_images = split_dataset(image_files)
        logger.info(f"数据集分割: 训练集 {len(train_images)}张, 验证集 {len(val_images)}张, 测试集 {len(test_images)}张")
    else:
        train_images = image_files
        val_images, test_images = [], []
    
    # 处理图像和标签
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split, images in splits.items():
        if not images:
            continue
            
        logger.info(f"处理{split}集...")
        
        for img_file in tqdm(images):
            img_path = os.path.join(train_img_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(train_label_dir, label_file)
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取图像: {img_path}")
                continue
                
            height, width = img.shape[:2]
            
            # 解析标签
            boxes = parse_yolo_label(label_path, width, height)
            if not boxes:
                logger.warning(f"图像没有有效标签: {img_path}")
                continue
            
            # 对每个标注框，裁剪并保存到相应类别目录
            for i, box in enumerate(boxes):
                class_id, x_min, y_min, x_max, y_max = box
                
                if class_id >= len(class_names):
                    logger.warning(f"类别ID超出范围: {class_id}, 最大ID应为 {len(class_names)-1}")
                    continue
                    
                class_name = class_names[class_id]
                
                # 计算像素坐标
                x_min_px = int(x_min * width)
                y_min_px = int(y_min * height)
                x_max_px = int(x_max * width)
                y_max_px = int(y_max * height)
                
                # 确保至少有1个像素的宽高
                if x_max_px <= x_min_px or y_max_px <= y_min_px:
                    logger.warning(f"无效的边界框大小: {[x_min_px, y_min_px, x_max_px, y_max_px]}")
                    continue
                
                # 裁剪图像
                try:
                    crop = img[y_min_px:y_max_px, x_min_px:x_max_px]
                    if crop.size == 0:
                        logger.warning(f"裁剪得到空图像: {[y_min_px, y_max_px, x_min_px, x_max_px]}")
                        continue
                except Exception as e:
                    logger.error(f"裁剪图像时出错: {e}")
                    continue
                
                # 保存裁剪的图像
                output_dir = os.path.join(output_path, split, class_name)
                output_file = f"{os.path.splitext(img_file)[0]}_crop{i}.jpg"
                output_path_full = os.path.join(output_dir, output_file)
                
                try:
                    cv2.imwrite(output_path_full, crop)
                except Exception as e:
                    logger.error(f"保存图像时出错 {output_path_full}: {e}")
    
    logger.info("CNN数据集创建完成")

def move_yolo_files(yolo_dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """将YOLO数据集按比例分割并移动文件
    
    Args:
        yolo_dataset_path: YOLO数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    train_img_dir = os.path.join(yolo_dataset_path, 'train', 'images')
    train_label_dir = os.path.join(yolo_dataset_path, 'train', 'labels')
    
    # 创建验证集和测试集目录
    for split in ['valid', 'test']:
        os.makedirs(os.path.join(yolo_dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_path, split, 'labels'), exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 分割数据集
    train_images, val_images, test_images = split_dataset(
        image_files, train_ratio, val_ratio, test_ratio)
    
    logger.info(f"YOLO数据集分割: 训练集 {len(train_images)}张, 验证集 {len(val_images)}张, 测试集 {len(test_images)}张")
    
    # 移动验证集文件
    for img_file in tqdm(val_images, desc="移动验证集文件"):
        label_file = os.path.splitext(img_file)[0] + '.txt'
        
        # 移动图像文件
        src_img = os.path.join(train_img_dir, img_file)
        dst_img = os.path.join(yolo_dataset_path, 'valid', 'images', img_file)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        
        # 移动标签文件
        src_label = os.path.join(train_label_dir, label_file)
        dst_label = os.path.join(yolo_dataset_path, 'valid', 'labels', label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
    
    # 移动测试集文件
    for img_file in tqdm(test_images, desc="移动测试集文件"):
        label_file = os.path.splitext(img_file)[0] + '.txt'
        
        # 移动图像文件
        src_img = os.path.join(train_img_dir, img_file)
        dst_img = os.path.join(yolo_dataset_path, 'test', 'images', img_file)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        
        # 移动标签文件
        src_label = os.path.join(train_label_dir, label_file)
        dst_label = os.path.join(yolo_dataset_path, 'test', 'labels', label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
    
    logger.info("YOLO数据集分割完成")

def main():
    """主函数"""
    # 读取YOLOv8的data.yaml获取类别信息
    yaml_path = os.path.join(args.yolo_data, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"错误：找不到YOLOv8数据集的data.yaml文件: {yaml_path}")
        return
    
    yaml_data = read_yaml_config(yaml_path)
    
    # 获取类别名称
    class_names = yaml_data.get('names', [])
    if not class_names:
        print("错误：data.yaml中未找到类别信息")
        return
    
    print(f"从data.yaml中读取到{len(class_names)}个类别:")
    for i, name in enumerate(class_names):
        print(f"  类别{i}: {name}")
    
    # 转换数据集
    create_cnn_dataset(
        args.yolo_data,
        args.output_dir,
        class_names,
        split_data=args.split_data
    )
    
    print("数据集转换完成！")
    print(f"CNN格式数据集已保存到: {args.output_dir}")
    
    # 检查各类别的样本数量
    print("\n各类别目录中的文件数量检查:")
    for phase in ['train', 'val', 'test']:
        phase_dir = os.path.join(args.output_dir, phase)
        if not os.path.exists(phase_dir):
            continue
            
        print(f"  {phase}:")
        for class_name in class_names:
            class_dir = os.path.join(phase_dir, class_name)
            if os.path.exists(class_dir):
                file_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"    {class_name}: {file_count}个图像文件")
            else:
                print(f"    {class_name}: 目录不存在")
    
    # 创建README文件
    readme_path = os.path.join(args.output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("CNN训练数据集\n")
        f.write("=============\n\n")
        f.write(f"该数据集由YOLOv8数据集({args.yolo_data})转换得到\n\n")
        f.write("类别信息:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
        f.write("\n")
        f.write("目录结构:\n")
        f.write("  train/  - 训练集\n")
        f.write("  val/    - 验证集\n")
        f.write("  test/   - 测试集（如果有）\n")
    
    print(f"已生成数据集说明文件: {readme_path}")


if __name__ == '__main__':
    main() 