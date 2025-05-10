#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统训练脚本
"""

import os
import argparse
import torch
from pathlib import Path
from models.yolo_model import YOLODetector
from models.cnn_model import CNNClassifier
from models.cascade_detector import CascadeDetector
from utils.data_utils import create_dataloaders, prepare_yolo_dataset, download_sample_dataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测系统训练脚本')
    
    # 数据集参数
    parser.add_argument('--data-dir', type=str, default='data/welding_defects',
                        help='数据集根目录')
    parser.add_argument('--download-data', action='store_true',
                        help='是否下载示例数据集结构')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--img-size', type=int, default=640,
                        help='YOLOv8训练图像大小')
    parser.add_argument('--cnn-size', type=int, default=224,
                        help='CNN训练图像大小')
    parser.add_argument('--yolo-epochs', type=int, default=100,
                        help='YOLOv8训练轮数')
    parser.add_argument('--cnn-epochs', type=int, default=50,
                        help='CNN训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='CNN学习率')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=4,
                        help='类别数量')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='YOLOv8置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='YOLOv8 IOU阈值')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/train',
                        help='输出目录')
    parser.add_argument('--yolo-only', action='store_true',
                        help='仅训练YOLOv8模型')
    parser.add_argument('--cnn-only', action='store_true',
                        help='仅训练CNN模型')
    
    return parser.parse_args()


def train_yolo(args):
    """训练YOLOv8模型"""
    print("=" * 50)
    print("开始训练YOLOv8模型")
    print("=" * 50)
    
    # 准备YOLO格式数据集
    class_names = ['normal', 'porosity', 'crack', 'slag']
    yolo_data_dir = os.path.join(args.output_dir, 'yolo_dataset')
    data_yaml_path = prepare_yolo_dataset(args.data_dir, yolo_data_dir, class_names)
    
    # 初始化YOLOv8模型
    model = YOLODetector()
    
    # 训练模型
    model.train(
        data_yaml=data_yaml_path,
        epochs=args.yolo_epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size
    )
    
    # 保存模型
    model_save_path = os.path.join(args.output_dir, 'yolo_model.pt')
    model.save(model_save_path)
    print(f"YOLOv8模型已保存到 {model_save_path}")
    
    return model_save_path


def train_cnn(args):
    """训练CNN模型"""
    print("=" * 50)
    print("开始训练CNN模型")
    print("=" * 50)
    
    # 创建数据加载器
    input_size = (args.cnn_size, args.cnn_size)
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        input_size=input_size
    )
    
    # 初始化CNN模型
    model = CNNClassifier(num_classes=args.num_classes, input_size=input_size)
    
    # 训练模型
    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.cnn_epochs,
        learning_rate=args.lr
    )
    
    # 保存模型
    model_save_path = os.path.join(args.output_dir, 'cnn_model.pt')
    model.save(model_save_path)
    print(f"CNN模型已保存到 {model_save_path}")
    
    return model_save_path


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 下载示例数据集（如果需要）
    if args.download_data:
        download_sample_dataset(args.data_dir)
    
    # 训练模型
    yolo_model_path = None
    cnn_model_path = None
    
    # 训练YOLOv8模型
    if not args.cnn_only:
        yolo_model_path = train_yolo(args)
    
    # 训练CNN模型
    if not args.yolo_only:
        cnn_model_path = train_cnn(args)
    
    # 如果两个模型都训练了，创建级联检测器并保存
    if yolo_model_path and cnn_model_path:
        print("=" * 50)
        print("创建级联检测器")
        print("=" * 50)
        
        detector = CascadeDetector(
            yolo_model_path=yolo_model_path,
            cnn_model_path=cnn_model_path,
            num_classes=args.num_classes,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
            cnn_input_size=(args.cnn_size, args.cnn_size)
        )
        
        print("级联检测器创建完成")


if __name__ == '__main__':
    main() 