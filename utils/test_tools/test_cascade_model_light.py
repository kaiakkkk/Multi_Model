#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统 - 级联模型测试脚本
用于验证数据处理并进行简单的训练测试
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import yaml
from pathlib import Path
import logging
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子，确保结果可重复
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SmallCNN(nn.Module):
    """简化版CNN模型，用于快速测试"""
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNNDataset(Dataset):
    """CNN数据集加载器"""
    def __init__(self, data_dir, transform=None, max_samples_per_class=100):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) 
                    if os.path.isfile(os.path.join(cls_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 限制每个类别的样本数量，加快测试速度
            if len(files) > max_samples_per_class:
                files = random.sample(files, max_samples_per_class)
            
            for file_path in files:
                self.samples.append((file_path, self.class_to_idx[cls_name]))
        
        logger.info(f"加载了{len(self.samples)}个样本，包含{len(self.classes)}个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class SimpleYOLODetector:
    """简化的YOLO检测器模拟"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def detect(self, image):
        """
        模拟YOLO检测过程，返回随机生成的边界框
        
        Args:
            image: 输入图像，numpy数组 (H, W, C)
        
        Returns:
            list of [x_min, y_min, x_max, y_max, conf, class_id]
        """
        height, width = image.shape[:2]
        num_boxes = random.randint(1, 3)  # 随机生成1-3个边界框
        
        boxes = []
        for _ in range(num_boxes):
            # 随机生成边界框
            x_min = random.uniform(0, 0.7) * width
            y_min = random.uniform(0, 0.7) * height
            box_width = random.uniform(0.2, 0.3) * width
            box_height = random.uniform(0.2, 0.3) * height
            x_max = min(x_min + box_width, width)
            y_max = min(y_min + box_height, height)
            
            # 随机生成置信度和类别
            conf = random.uniform(0.5, 0.9)
            class_id = random.randint(0, 4)
            
            if conf > self.threshold:
                boxes.append([int(x_min), int(y_min), int(x_max), int(y_max), conf, class_id])
        
        return boxes

class CascadeDetector:
    """级联检测器：YOLO+CNN"""
    def __init__(self, yolo_detector, cnn_model, class_names, transform):
        self.yolo_detector = yolo_detector
        self.cnn_model = cnn_model
        self.class_names = class_names
        self.transform = transform
    
    def detect(self, image):
        """
        使用级联检测
        
        Args:
            image: 输入图像，numpy数组 (H, W, C)
        
        Returns:
            list of [x_min, y_min, x_max, y_max, conf, label]
        """
        # YOLO阶段检测
        yolo_boxes = self.yolo_detector.detect(image)
        
        refined_results = []
        for box in yolo_boxes:
            x_min, y_min, x_max, y_max, conf, _ = box
            
            # 裁剪图像区域
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
                
            # 将裁剪区域转换为CNN输入
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_tensor = self.transform(crop_pil).unsqueeze(0).to(device)
            
            # CNN模型预测
            with torch.no_grad():
                outputs = self.cnn_model(crop_tensor)
                _, predicted = torch.max(outputs, 1)
                
                # 获取CNN预测的类别和置信度
                cnn_class = predicted.item()
                cnn_conf = torch.softmax(outputs, dim=1)[0, cnn_class].item()
                
                # 结合两个模型的置信度
                final_conf = conf * cnn_conf
                
                refined_results.append([x_min, y_min, x_max, y_max, final_conf, cnn_class])
        
        return refined_results

def train_cnn(model, train_loader, val_loader, num_epochs=5):
    """
    训练CNN模型
    
    Args:
        model: CNN模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
    
    Returns:
        训练好的模型
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        logger.info(f"Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        logger.info(f"Validation Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    return model

def test_cascade_detector(cascade_detector, test_images, output_dir):
    """
    测试级联检测器
    
    Args:
        cascade_detector: 级联检测器
        test_images: 测试图像路径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_path in enumerate(test_images):
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"无法读取图像: {img_path}")
            continue
        
        # 级联检测
        results = cascade_detector.detect(image)
        
        # 可视化结果
        for box in results:
            x_min, y_min, x_max, y_max, conf, class_id = box
            
            label = cascade_detector.class_names[class_id]
            color = (0, 255, 0)  # 绿色边框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # 添加标签
            text = f"{label}: {conf:.2f}"
            cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"result_{i}.jpg")
        cv2.imwrite(output_path, image)
    
    logger.info(f"测试完成，结果保存在: {output_dir}")

def prepare_small_test_dataset(src_dir, dst_dir, samples_per_class=10):
    """
    准备一个小型测试数据集
    
    Args:
        src_dir: 源数据目录
        dst_dir: 目标数据目录
        samples_per_class: 每个类别选择的样本数
    """
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    
    # 查找所有图像文件
    image_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # 随机选择样本
    if len(image_files) > samples_per_class:
        selected_files = random.sample(image_files, samples_per_class)
    else:
        selected_files = image_files
    
    # 复制文件
    for i, file_path in enumerate(selected_files):
        dst_file = os.path.join(dst_dir, f"test_{i}.jpg")
        shutil.copy(file_path, dst_file)
    
    logger.info(f"准备了{len(selected_files)}个测试样本")
    return [os.path.join(dst_dir, f"test_{i}.jpg") for i in range(len(selected_files))]

def main():
    """主函数"""
    # 设置路径
    cnn_data_root = 'data/welding_defects/cnn'
    yolo_data_yaml = 'data/welding_defects/yolov8/data.yaml'
    output_dir = 'results/test_cascade'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取类别名称
    with open(yolo_data_yaml, 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']
    num_classes = len(class_names)
    
    logger.info(f"类别名称: {class_names}")
    
    # 定义数据转换
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 加载数据集，使用少量数据进行快速测试
    train_dataset = CNNDataset(
        os.path.join(cnn_data_root, 'train'),
        transform=data_transform['train'],
        max_samples_per_class=50  # 每个类别最多使用50个样本
    )
    
    val_dataset = CNNDataset(
        os.path.join(cnn_data_root, 'val'),
        transform=data_transform['val'],
        max_samples_per_class=20  # 每个类别最多使用20个样本
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    logger.info("数据加载完成，开始训练CNN模型")
    
    # 创建CNN模型
    model = SmallCNN(num_classes).to(device)
    
    # 训练模型 (仅做简单训练，快速测试级联系统)
    model = train_cnn(model, train_loader, val_loader, num_epochs=3)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
    logger.info(f"CNN模型已保存: {os.path.join(output_dir, 'cnn_model.pth')}")
    
    # 创建YOLO检测器 (使用简化版本进行测试)
    yolo_detector = SimpleYOLODetector(threshold=0.5)
    
    # 创建级联检测器
    cascade_detector = CascadeDetector(
        yolo_detector=yolo_detector,
        cnn_model=model,
        class_names=class_names,
        transform=data_transform['val']
    )
    
    # 准备测试图像
    test_dir = os.path.join(output_dir, 'test_images')
    test_images = prepare_small_test_dataset(
        os.path.join(cnn_data_root, 'test'),
        test_dir,
        samples_per_class=3
    )
    
    if not test_images:
        logger.warning("没有找到测试图像")
        return
    
    # 测试级联检测器
    logger.info("开始测试级联检测器")
    test_cascade_detector(cascade_detector, test_images, os.path.join(output_dir, 'detections'))
    
    logger.info("级联系统测试完成")

if __name__ == '__main__':
    main() 