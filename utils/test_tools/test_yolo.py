#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统 - YOLOv8测试脚本
用于验证YOLOv8模型的检测效果
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import yaml
import logging
import random
import shutil
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_class_names(yaml_path):
    """
    从YAML文件加载类别名称
    
    Args:
        yaml_path: YAML文件路径
    
    Returns:
        类别名称列表
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def prepare_test_images(yolo_data_path, output_dir, samples_count=5):
    """
    准备测试图像
    
    Args:
        yolo_data_path: YOLO数据集路径
        output_dir: 输出目录
        samples_count: 测试样本数量
        
    Returns:
        测试图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 优先使用测试集，如果没有则使用验证集
    test_img_dir = os.path.join(yolo_data_path, 'test', 'images')
    if not os.path.exists(test_img_dir):
        test_img_dir = os.path.join(yolo_data_path, 'valid', 'images')
        if not os.path.exists(test_img_dir):
            # 如果都没有，使用训练集
            test_img_dir = os.path.join(yolo_data_path, 'train', 'images')
    
    if not os.path.exists(test_img_dir):
        logger.error(f"找不到图像目录: {test_img_dir}")
        return []
    
    # 获取所有图像文件
    img_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        logger.error(f"目录中没有图像文件: {test_img_dir}")
        return []
    
    # 随机选择样本
    if len(img_files) > samples_count:
        selected_files = random.sample(img_files, samples_count)
    else:
        selected_files = img_files
    
    # 复制文件到输出目录
    test_images = []
    for i, file in enumerate(selected_files):
        src_path = os.path.join(test_img_dir, file)
        dst_path = os.path.join(output_dir, f"test_{i+1}.jpg")
        shutil.copy(src_path, dst_path)
        test_images.append(dst_path)
        logger.info(f"已准备测试图像: {dst_path}")
    
    return test_images

def test_yolo_detection(model, test_images, class_names, output_dir, conf_threshold=0.25):
    """
    测试YOLOv8检测
    
    Args:
        model: YOLOv8模型
        test_images: 测试图像路径列表
        class_names: 类别名称列表
        output_dir: 输出目录
        conf_threshold: 置信度阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in test_images:
        # 使用YOLO进行检测
        results = model(img_path, conf=conf_threshold)
        
        # 获取第一个结果
        result = results[0]
        
        # 原始图像
        img = cv2.imread(img_path)
        
        # 获取预测结果
        boxes = result.boxes
        
        # 检测到的目标数量
        logger.info(f"图像 {os.path.basename(img_path)} 检测到 {len(boxes)} 个目标")
        
        # 绘制检测结果
        for box in boxes:
            # 边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 置信度
            conf = float(box.conf[0])
            
            # 类别
            cls_id = int(box.cls[0])
            if cls_id < len(class_names):
                cls_name = class_names[cls_id]
            else:
                cls_name = f"未知-{cls_id}"
                logger.warning(f"类别索引 {cls_id} 超出范围 (0-{len(class_names)-1})")
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"yolo_result_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img)
        logger.info(f"保存检测结果: {output_path}")
    
    logger.info(f"检测完成，结果保存在: {output_dir}")

def test_pretrained_yolo():
    """测试预训练的YOLOv8模型"""
    # 使用预训练的YOLOv8n模型
    model = YOLO("yolov8n.pt")
    
    # 准备测试图像
    output_dir = "results/yolo_test"
    os.makedirs(output_dir, exist_ok=True)
    
    test_img_dir = os.path.join(output_dir, "test_images")
    os.makedirs(test_img_dir, exist_ok=True)
    
    # 使用COCO数据集的类别名称（完整80类）
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]
    
    # 生成随机图像用于测试
    img_urls = [
        "https://ultralytics.com/images/zidane.jpg",
        "https://ultralytics.com/images/bus.jpg"
    ]
    
    test_images = []
    for i, url in enumerate(img_urls):
        # 使用cv2从网络下载图像
        import urllib.request
        try:
            with urllib.request.urlopen(url) as resp:
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                
                img_path = os.path.join(test_img_dir, f"test_{i+1}.jpg")
                cv2.imwrite(img_path, img)
                test_images.append(img_path)
                logger.info(f"已下载测试图像: {img_path}")
        except Exception as e:
            logger.error(f"下载图像失败 {url}: {e}")
    
    # 测试检测
    detection_output_dir = os.path.join(output_dir, "detections")
    test_yolo_detection(model, test_images, class_names, detection_output_dir)

def main():
    """主函数"""
    # 检查是否有自定义模型路径
    if len(sys.argv) > 1 and sys.argv[1] == "--pretrained":
        logger.info("使用预训练YOLOv8模型进行测试")
        test_pretrained_yolo()
        return
    
    # 设置路径
    yolo_data_path = 'data/welding_defects/yolov8'
    yaml_path = os.path.join(yolo_data_path, 'data.yaml')
    output_dir = 'results/yolo_test'
    
    # 检查YAML文件是否存在
    if not os.path.exists(yaml_path):
        logger.error(f"找不到YAML文件: {yaml_path}")
        return
    
    # 加载类别名称
    class_names = load_class_names(yaml_path)
    logger.info(f"类别名称: {class_names}")
    
    # 检查YOLOv8模型
    model_path = 'models/yolov8n.pt'  # 预训练模型
    custom_model = 'models/welding_yolov8n.pt'  # 自定义模型
    
    if os.path.exists(custom_model):
        logger.info(f"使用自定义模型: {custom_model}")
        model_path = custom_model
    else:
        logger.info(f"使用预训练模型: {model_path}")
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info(f"下载YOLOv8n预训练模型...")
            # 这里使用YOLO直接下载预训练模型
            model = YOLO("yolov8n.pt")
            # 保存模型
            model.export(format="pt")
            shutil.copy("yolov8n.pt", model_path)
            logger.info(f"模型已保存: {model_path}")
        else:
            # 加载模型
            model = YOLO(model_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备测试图像
    test_img_dir = os.path.join(output_dir, "test_images")
    test_images = prepare_test_images(yolo_data_path, test_img_dir, samples_count=5)
    
    if not test_images:
        logger.error("没有可用的测试图像")
        return
    
    # 测试YOLOv8检测
    detection_output_dir = os.path.join(output_dir, "detections")
    test_yolo_detection(model, test_images, class_names, detection_output_dir)
    
    logger.info("YOLOv8测试完成")

if __name__ == '__main__':
    main() 