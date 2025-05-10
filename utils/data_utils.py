import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import yaml
import shutil

class WeldingDefectDataset(Dataset):
    """焊缝缺陷数据集类"""
    
    def __init__(self, data_dir, transform=None, phase='train'):
        """
        初始化焊缝缺陷数据集
        
        参数:
            data_dir: 数据集根目录
            transform: 图像变换
            phase: 'train', 'val' 或 'test'
        """
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform
        
        # 类别映射
        self.class_names = ['normal', 'porosity', 'crack', 'slag']
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # 获取图像文件路径和标签
        self.samples = self._load_samples()
        
        print(f"加载了 {len(self.samples)} 个 {phase} 样本")
    
    def _load_samples(self):
        """加载样本路径和标签"""
        samples = []
        
        # 构建相对路径
        phase_dir = os.path.join(self.data_dir, self.phase)
        
        # 遍历每个类别目录
        for class_name in self.class_names:
            class_dir = os.path.join(phase_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"警告: 类别目录 {class_dir} 不存在")
                continue
            
            # 获取此类别的所有图像
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    samples.append((img_path, label))
        
        return samples
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        img_path, label = self.samples[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (224, 224), color='black')
            
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def create_dataloaders(data_dir, batch_size=16, input_size=(224, 224), num_workers=4):
    """
    创建数据加载器
    
    参数:
        data_dir: 数据集根目录
        batch_size: 批大小
        input_size: 输入图像尺寸 (高度, 宽度)
        num_workers: 数据加载线程数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证/测试集变换
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = WeldingDefectDataset(data_dir, transform=train_transform, phase='train')
    val_dataset = WeldingDefectDataset(data_dir, transform=val_transform, phase='val')
    test_dataset = WeldingDefectDataset(data_dir, transform=val_transform, phase='test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def prepare_yolo_dataset(data_dir, output_dir, class_names):
    """
    准备YOLOv8格式的数据集
    
    参数:
        data_dir: 源数据目录
        output_dir: 输出目录
        class_names: 类别名称列表
    
    返回:
        data_yaml_path: YAML配置文件路径
    """
    # 创建YOLO数据集目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建images和labels目录
    for phase in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, phase, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, phase, 'labels'), exist_ok=True)
    
    # 遍历数据集进行转换
    class_indices = {name: i for i, name in enumerate(class_names)}
    
    for phase in ['train', 'val', 'test']:
        phase_dir = os.path.join(data_dir, phase)
        
        if not os.path.exists(phase_dir):
            print(f"目录 {phase_dir} 不存在，跳过")
            continue
        
        # 遍历每个类别
        for class_name in class_names:
            class_dir = os.path.join(phase_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"目录 {class_dir} 不存在，跳过")
                continue
            
            # 处理每张图像
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # 源文件路径
                    img_path = os.path.join(class_dir, img_name)
                    
                    # 复制图像到YOLO目录
                    dest_img_path = os.path.join(output_dir, phase, 'images', img_name)
                    shutil.copy(img_path, dest_img_path)
                    
                    # 读取图像获取尺寸
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"无法读取图像 {img_path}")
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # 创建标签文件 (使用与图像相同的名称，但扩展名为.txt)
                    label_name = os.path.splitext(img_name)[0] + '.txt'
                    label_path = os.path.join(output_dir, phase, 'labels', label_name)
                    
                    # 假设整个图像都是缺陷区域
                    # YOLO格式: <class_id> <center_x> <center_y> <width> <height>, 所有值都是相对于图像尺寸归一化
                    class_id = class_indices[class_name]
                    center_x, center_y = 0.5, 0.5  # 中心点 (归一化)
                    width, height = 0.9, 0.9      # 宽高 (归一化)
                    
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
    
    # 创建data.yaml文件
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"YOLO数据集准备完成，配置文件保存在: {data_yaml_path}")
    return data_yaml_path


def download_sample_dataset(output_dir):
    """
    下载示例焊缝缺陷数据集
    
    参数:
        output_dir: 输出目录
    """
    print("注意: 这只是一个模拟函数，实际实现需要用户根据自己的数据集情况修改")
    print(f"将创建模拟数据集目录结构在: {output_dir}")
    
    # 创建目录结构
    for phase in ['train', 'val', 'test']:
        for cls in ['normal', 'porosity', 'crack', 'slag']:
            os.makedirs(os.path.join(output_dir, phase, cls), exist_ok=True)
    
    print("目录结构已创建，请将对应的焊缝缺陷图片放入相应目录中")
    print("或者访问以下推荐的公开焊缝缺陷数据集:")
    print("1. GDXray+ 数据集: https://github.com/computervision-xray-testing/GDXray")
    print("2. KolektorSDD2 数据集: https://www.vicos.si/resources/kolektorsdd2/")
    print("3. NEU表面缺陷数据集: http://faculty.neu.edu.cn/songkc/en/su.html")
    
    # 创建一个简单的README文件
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("焊缝缺陷数据集目录结构\n")
        f.write("------------------------\n\n")
        f.write("数据集应按以下结构组织:\n")
        f.write("data_root/\n")
        f.write("  ├── train/\n")
        f.write("  │   ├── normal/\n")
        f.write("  │   ├── porosity/\n")
        f.write("  │   ├── crack/\n")
        f.write("  │   └── slag/\n")
        f.write("  ├── val/\n")
        f.write("  │   ├── normal/\n")
        f.write("  │   ├── porosity/\n")
        f.write("  │   ├── crack/\n")
        f.write("  │   └── slag/\n")
        f.write("  └── test/\n")
        f.write("      ├── normal/\n")
        f.write("      ├── porosity/\n")
        f.write("      ├── crack/\n")
        f.write("      └── slag/\n")
    
    print(f"已创建README文件: {readme_path}") 