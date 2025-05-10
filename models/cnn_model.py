import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class CNNClassifier:
    """基于ResNet50的CNN分类器类，用于焊缝缺陷的精细分类"""
    
    def __init__(self, num_classes=4, model_path=None, input_size=(224, 224)):
        """
        初始化CNN分类器
        
        参数:
            num_classes: 缺陷类别数量
            model_path: 预训练模型路径，如果为None则从头训练
            input_size: 输入图像尺寸(高度,宽度)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        # 创建ResNet50模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 修改最后的全连接层以适应我们的分类任务
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 将模型移动到设备上
        self.model = self.model.to(self.device)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 如果提供了模型路径，则加载预训练权重
        if model_path:
            self.load(model_path)
        
        print(f"CNN分类器已初始化，使用设备: {self.device}")
    
    def preprocess(self, image):
        """
        预处理输入图像
        
        参数:
            image: 输入图像(OpenCV格式BGR)
            
        返回:
            tensor: 预处理后的图像张量
        """
        # OpenCV格式(BGR)转换为PIL格式(RGB)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # 应用变换
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # 添加批次维度
        tensor = tensor.to(self.device)
        
        return tensor
    
    def classify(self, image):
        """
        对输入图像进行分类
        
        参数:
            image: 输入图像(OpenCV BGR格式或PIL格式)
            
        返回:
            class_id: 预测的类别ID
            confidence: 置信度
        """
        # 图像预处理
        tensor = self.preprocess(image)
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted = torch.max(probs, 1)
            
            class_id = predicted.item()
            confidence = confidence.item()
        
        return class_id, confidence
    
    def train(self, train_loader, val_loader=None, epochs=50, learning_rate=0.001):
        """
        训练CNN模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            
            # 验证
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
                scheduler.step(val_loss)
    
    def evaluate(self, data_loader, criterion=None):
        """评估模型性能"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        eval_loss = running_loss / len(data_loader)
        eval_acc = 100 * correct / total
        
        return eval_loss, eval_acc
    
    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 {path} 加载")
        return self 