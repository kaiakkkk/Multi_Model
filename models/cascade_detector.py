import torch
import cv2
import numpy as np
from models.yolo_model import YOLODetector
from models.cnn_model import CNNClassifier

class CascadeDetector:
    """级联焊缝缺陷检测器，结合YOLOv8和CNN模型"""
    
    def __init__(self, yolo_model_path=None, cnn_model_path=None, num_classes=4,
                 conf_threshold=0.25, iou_threshold=0.45, cnn_input_size=(224, 224)):
        """
        初始化级联检测器
        
        参数:
            yolo_model_path: YOLOv8模型路径
            cnn_model_path: CNN模型路径
            num_classes: 类别数量
            conf_threshold: YOLOv8置信度阈值
            iou_threshold: YOLOv8 IOU阈值
            cnn_input_size: CNN输入图像尺寸
        """
        # 初始化YOLOv8检测器
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 初始化CNN分类器
        self.cnn_classifier = CNNClassifier(
            num_classes=num_classes,
            model_path=cnn_model_path,
            input_size=cnn_input_size
        )
        
        # 缺陷类别名称
        self.class_names = ['正常', '气孔', '裂纹', '夹渣']
        
        print("级联检测器已初始化")
    
    def detect(self, image):
        """
        对输入图像进行级联检测
        
        参数:
            image: 输入图像(OpenCV BGR格式)
            
        返回:
            results: 包含检测结果的列表，每个元素为[x1, y1, x2, y2, yolo_conf, yolo_class_id, cnn_class_id, cnn_conf]
        """
        # 第一阶段：YOLOv8检测
        bboxes, crops = self.yolo_detector.detect(image)
        
        if not bboxes:
            return []
        
        # 第二阶段：CNN精细分类
        refined_results = []
        
        for i, (bbox, crop) in enumerate(zip(bboxes, crops)):
            x1, y1, x2, y2, yolo_conf, yolo_class_id = bbox
            
            # 确保裁剪区域有效
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            
            # CNN分类
            cnn_class_id, cnn_conf = self.cnn_classifier.classify(crop)
            
            # 合并结果
            refined_results.append([
                x1, y1, x2, y2, 
                yolo_conf, 
                yolo_class_id,
                cnn_class_id, 
                cnn_conf
            ])
        
        return refined_results
    
    def visualize(self, image, results):
        """
        可视化检测结果
        
        参数:
            image: 输入图像
            results: 检测结果
            
        返回:
            vis_image: 带有检测框和标签的图像
        """
        vis_image = image.copy()
        
        # 颜色映射
        colors = [
            (0, 255, 0),    # 正常 - 绿色
            (0, 165, 255),  # 气孔 - 橙色
            (0, 0, 255),    # 裂纹 - 红色
            (255, 0, 0)     # 夹渣 - 蓝色
        ]
        
        for res in results:
            x1, y1, x2, y2, yolo_conf, yolo_class_id, cnn_class_id, cnn_conf = res
            
            # 获取颜色和类别名称
            color = colors[cnn_class_id % len(colors)]
            label = f"{self.class_names[cnn_class_id]}: {cnn_conf:.2f}"
            
            # 绘制矩形框
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制标签背景
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                vis_image, 
                (int(x1), int(y1) - text_size[1] - 5),
                (int(x1) + text_size[0], int(y1)), 
                color, 
                -1
            )
            
            # 绘制标签
            cv2.putText(
                vis_image, 
                label, 
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        return vis_image
    
    def save_models(self, yolo_path, cnn_path):
        """保存模型"""
        self.yolo_detector.save(yolo_path)
        self.cnn_classifier.save(cnn_path)
    
    def load_models(self, yolo_path, cnn_path):
        """加载模型"""
        self.yolo_detector.load(yolo_path)
        self.cnn_classifier.load(cnn_path) 