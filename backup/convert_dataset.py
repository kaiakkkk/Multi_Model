#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统数据集转换脚本
将YOLOv8格式数据集转换为CNN训练所需的格式
"""

import os
import argparse
from utils.convert_yolo_to_cnn import read_yaml_config, create_cnn_dataset, move_yolo_files
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO多边形标注转换为CNN数据集')
    
    parser.add_argument('--yolo-path', type=str, default='data/welding_defects/yolov8',
                        help='YOLO数据集路径 (默认: data/welding_defects/yolov8)')
    
    parser.add_argument('--output-path', type=str, default='data/welding_defects/cnn',
                        help='CNN数据集输出路径 (默认: data/welding_defects/cnn)')
    
    parser.add_argument('--split-yolo', action='store_true',
                        help='是否将YOLO数据集按比例分割 (默认: False)')
    
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例 (默认: 0.7)')
    
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 检查YOLO数据集路径
    if not os.path.exists(args.yolo_path):
        logger.error(f"YOLO数据集路径不存在: {args.yolo_path}")
        return
    
    # 检查YAML配置文件
    yaml_path = os.path.join(args.yolo_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        logger.error(f"YAML配置文件不存在: {yaml_path}")
        return
    
    # 读取YAML配置
    config = read_yaml_config(yaml_path)
    class_names = config.get('names', [])
    if not class_names:
        logger.error("YAML配置文件中没有找到类别名称")
        return
    
    logger.info(f"类别名称: {class_names}")
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 如果需要分割YOLO数据集
    if args.split_yolo:
        logger.info("正在分割YOLO数据集...")
        move_yolo_files(
            args.yolo_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    # 创建CNN数据集
    logger.info("正在创建CNN数据集...")
    create_cnn_dataset(
        args.yolo_path,
        args.output_path,
        class_names,
        split_data=True
    )
    
    logger.info(f"CNN数据集已创建: {args.output_path}")
    
    # 打印类别统计信息
    if os.path.exists(args.output_path):
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(args.output_path, split)
            if os.path.exists(split_dir):
                logger.info(f"{split}集目录统计:")
                for class_name in class_names:
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.exists(class_dir):
                        file_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        logger.info(f"  - {class_name}: {file_count}张图像")

if __name__ == '__main__':
    main() 