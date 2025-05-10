#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统 - 级联系统简化测试脚本
使用少量数据测试YOLO+CNN级联检测效果
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
from ultralytics import YOLO
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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
    torch.backends.cudnn.deterministic = True

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleDataset(Dataset):
    """简单数据集加载器"""
    def __init__(self, root_dir, split='train', transform=None, max_samples=50):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.max_samples = max_samples
        
        self.classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # 限制每个类别的样本数
            if len(files) > max_samples:
                files = random.sample(files, max_samples)
                
            for f in files:
                self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))
        
        logger.info(f"加载了 {len(self.samples)} 个样本，{len(self.classes)} 个类别")
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SimpleCNN(nn.Module):
    """简化版CNN模型，用于测试"""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 使用预训练的ResNet18作为基础模型
        self.model = models.resnet18(pretrained=True)
        # 修改最后的全连接层以适应我们的类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class CascadeDetector:
    """级联检测器：YOLO+CNN"""
    def __init__(self, yolo_model, cnn_model, class_names, transform, conf_threshold=0.25):
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.class_names = class_names
        self.transform = transform
        self.conf_threshold = conf_threshold
        
    def detect(self, image_path):
        """
        对图像进行级联检测
        
        Args:
            image_path: 图像路径
            
        Returns:
            原始图像和检测结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None, []
            
        # 第一阶段：YOLO检测
        results = self.yolo_model(image_path, conf=self.conf_threshold)
        result = results[0]
        
        # 获取预测的边界框
        boxes = result.boxes
        logger.info(f"YOLO检测到 {len(boxes)} 个目标")
        
        # 如果没有检测到目标，直接返回
        if len(boxes) == 0:
            return image, []
            
        # 第二阶段：CNN分类
        refined_results = []
        
        for box in boxes:
            # 边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # YOLO的置信度
            yolo_conf = float(box.conf[0])
            
            # 裁剪目标区域
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # 预处理
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(device)
            
            # CNN预测
            with torch.no_grad():
                self.cnn_model.eval()
                outputs = self.cnn_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                class_id = predicted.item()
                cnn_conf = confidence.item()
                
                # 计算最终置信度
                final_conf = yolo_conf * cnn_conf
                
                # 添加结果
                refined_results.append({
                    'box': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'yolo_conf': yolo_conf,
                    'cnn_conf': cnn_conf,
                    'final_conf': final_conf
                })
                
                logger.debug(f"检测结果: 类别={self.class_names[class_id]}, YOLO置信度={yolo_conf:.4f}, CNN置信度={cnn_conf:.4f}, 最终置信度={final_conf:.4f}")
        
        return image, refined_results

def train_small_cnn(data_root, output_path, num_epochs=3, batch_size=16):
    """
    训练一个小型CNN模型用于级联系统测试
    
    Args:
        data_root: 数据集根目录
        output_path: 模型保存路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        
    Returns:
        训练好的模型和类别名称
    """
    # 定义数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = SimpleDataset(data_root, 'train', train_transform, max_samples=50)
    val_dataset = SimpleDataset(data_root, 'val', val_transform, max_samples=20)
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        logger.error("训练集为空")
        return None, []
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    model = SimpleCNN(len(train_dataset.classes)).to(device)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    logger.info("开始训练CNN模型...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
        
        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total
        
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
        
        val_loss = running_loss / len(val_dataset)
        val_acc = correct / total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"模型已保存: {output_path}")
    
    return model, train_dataset.classes

def prepare_test_images(yolo_data_path, output_dir, count=5):
    """
    准备测试图像
    
    Args:
        yolo_data_path: YOLO数据集路径
        output_dir: 输出目录
        count: 测试图像数量
        
    Returns:
        测试图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先尝试使用测试集
    test_dir = os.path.join(yolo_data_path, 'test', 'images')
    if not os.path.exists(test_dir):
        # 如果测试集不存在，尝试验证集
        test_dir = os.path.join(yolo_data_path, 'valid', 'images')
        if not os.path.exists(test_dir):
            # 如果验证集也不存在，使用训练集
            test_dir = os.path.join(yolo_data_path, 'train', 'images')
    
    if not os.path.exists(test_dir):
        logger.error(f"找不到图像目录: {test_dir}")
        return []
    
    # 获取所有图像文件
    img_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        logger.error(f"目录中没有图像文件: {test_dir}")
        return []
    
    # 随机选择图像
    if len(img_files) > count:
        selected_files = random.sample(img_files, count)
    else:
        selected_files = img_files
    
    # 复制图像到输出目录
    test_images = []
    for i, file in enumerate(selected_files):
        src_path = os.path.join(test_dir, file)
        dst_path = os.path.join(output_dir, f"test_{i+1}.jpg")
        shutil.copy(src_path, dst_path)
        test_images.append(dst_path)
        logger.info(f"已准备测试图像: {dst_path}")
    
    return test_images

def visualize_detection(image, results, class_names, output_path):
    """
    可视化检测结果
    
    Args:
        image: 原始图像 
        results: 检测结果列表
        class_names: 类别名称列表
        output_path: 输出路径
    """
    # 绘制检测结果
    for result in results:
        x1, y1, x2, y2 = result['box']
        class_id = result['class_id']
        final_conf = result['final_conf']
        
        # 类别名称
        if class_id < len(class_names):
            cls_name = class_names[class_id]
        else:
            cls_name = f"未知-{class_id}"
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签
        label = f"{cls_name}: {final_conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(output_path, image)
    logger.info(f"检测结果已保存: {output_path}")

def main():
    """主函数"""
    # 设置路径
    data_root = 'data/welding_defects'
    cnn_data_root = os.path.join(data_root, 'cnn')
    yolo_data_path = os.path.join(data_root, 'yolov8')
    yaml_path = os.path.join(yolo_data_path, 'data.yaml')
    output_dir = 'results/cascade_simple'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查YAML文件是否存在
    if not os.path.exists(yaml_path):
        logger.error(f"找不到YAML文件: {yaml_path}")
        return
    
    # 读取类别名称
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    logger.info(f"类别名称: {class_names}")
    
    # 设置CNN模型路径
    cnn_model_path = os.path.join(output_dir, 'cnn_model.pth')
    
    # 训练CNN模型
    if os.path.exists(cnn_model_path):
        logger.info(f"加载已有CNN模型: {cnn_model_path}")
        cnn_model = SimpleCNN(len(class_names)).to(device)
        cnn_model.load_state_dict(torch.load(cnn_model_path))
    else:
        logger.info("训练新的CNN模型...")
        cnn_model, _ = train_small_cnn(cnn_data_root, cnn_model_path, num_epochs=3)
        if cnn_model is None:
            logger.error("CNN模型训练失败")
            return
    
    # 加载YOLO模型
    yolo_model_path = os.path.join('models', 'yolov8n.pt')
    if not os.path.exists(yolo_model_path):
        os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
        logger.info("下载YOLOv8预训练模型...")
        yolo_model = YOLO('yolov8n.pt')
    else:
        logger.info(f"加载YOLOv8模型: {yolo_model_path}")
        yolo_model = YOLO(yolo_model_path)
    
    # 设置推理变换
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建级联检测器
    cascade_detector = CascadeDetector(
        yolo_model=yolo_model,
        cnn_model=cnn_model,
        class_names=class_names,
        transform=inference_transform,
        conf_threshold=0.25
    )
    
    # 准备测试图像
    test_img_dir = os.path.join(output_dir, 'test_images')
    test_images = prepare_test_images(yolo_data_path, test_img_dir, count=5)
    
    if not test_images:
        logger.error("没有可用的测试图像")
        return
    
    # 进行级联检测
    logger.info("开始级联检测...")
    detection_output_dir = os.path.join(output_dir, 'detections')
    os.makedirs(detection_output_dir, exist_ok=True)
    
    for i, img_path in enumerate(test_images):
        logger.info(f"处理图像 {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        
        # 级联检测
        image, results = cascade_detector.detect(img_path)
        
        if image is None:
            logger.error(f"处理图像失败: {img_path}")
            continue
            
        # 可视化结果
        output_path = os.path.join(detection_output_dir, f"cascade_result_{os.path.basename(img_path)}")
        visualize_detection(image, results, class_names, output_path)
    
    logger.info("级联检测完成！")
    logger.info(f"结果保存在: {detection_output_dir}")

if __name__ == '__main__':
    main() 