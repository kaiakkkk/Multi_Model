from ultralytics import YOLO
import torch
import numpy as np
import cv2

class YOLODetector:
    """YOLOv8焊缝缺陷检测模型类"""
    
    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化YOLOv8检测器
        
        参数:
            model_path: YOLOv8模型路径，如果为None则加载预训练模型
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 默认使用YOLOv8n模型
            self.model = YOLO('yolov8n.pt')
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"YOLOv8检测器已初始化，使用设备: {self.device}")
    
    def detect(self, image):
        """
        对输入图像进行缺陷检测，返回检测框
        
        参数:
            image: 输入图像，OpenCV格式(BGR)
            
        返回:
            bboxes: 边界框列表，每个元素为[x1, y1, x2, y2, conf, class_id]
            crops: 裁剪的候选区域图像列表
        """
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # 获取检测结果
        bboxes = []
        crops = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # 获取位置信息 [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 获取置信度和类别
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # 存储结果
                bboxes.append([x1, y1, x2, y2, conf, cls_id])
                
                # 裁剪检测区域
                crop = image[y1:y2, x1:x2]
                crops.append(crop)
        
        return bboxes, crops
    
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640):
        """
        训练YOLOv8模型
        
        参数:
            data_yaml: 数据配置文件路径
            epochs: 训练轮数
            batch_size: 批量大小
            imgsz: 图像大小
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
    
    def save(self, path):
        """保存模型"""
        self.model.save(path)
    
    def load(self, path):
        """加载模型"""
        self.model = YOLO(path)
        return self 