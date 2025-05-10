#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统可视化脚本
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from models.cascade_detector import CascadeDetector


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测系统可视化脚本')
    
    # 输入参数
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--result', type=str, default=None,
                        help='检测结果文本文件路径，如果为None则重新执行检测')
    
    # 模型参数
    parser.add_argument('--yolo-model', type=str, default='runs/train/yolo_model.pt',
                        help='YOLOv8模型路径')
    parser.add_argument('--cnn-model', type=str, default='runs/train/cnn_model.pt',
                        help='CNN模型路径')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IOU阈值')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='类别数量')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/visualize',
                        help='输出目录')
    
    return parser.parse_args()


def load_results(result_path):
    """
    从文本文件加载检测结果
    
    参数:
        result_path: 结果文件路径
        
    返回:
        results: 检测结果列表
    """
    results = []
    
    with open(result_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 8:  # 确保格式正确
                # 格式: <x1> <y1> <x2> <y2> <yolo_conf> <yolo_class> <cnn_class> <cnn_conf>
                x1, y1, x2, y2 = map(float, values[:4])
                yolo_conf = float(values[4])
                yolo_cls = int(values[5])
                cnn_cls = int(values[6])
                cnn_conf = float(values[7])
                
                results.append([x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf])
    
    return results


def visualize_defect_types(image, results, output_path, class_names=None):
    """
    可视化不同类型的缺陷及其数量分布
    
    参数:
        image: 输入图像
        results: 检测结果
        output_path: 输出路径
        class_names: 类别名称
    """
    if class_names is None:
        class_names = ['正常', '气孔', '裂纹', '夹渣']
    
    # 统计每种类型的数量
    class_counts = {}
    for res in results:
        cnn_cls = int(res[6])
        class_name = class_names[cnn_cls] if cnn_cls < len(class_names) else f"未知类别{cnn_cls}"
        
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    # 创建可视化图表
    plt.figure(figsize=(12, 8))
    
    # 图1：检测原图
    plt.subplot(2, 2, 1)
    vis_image = image.copy()
    
    # 颜色映射
    colors = [
        (0, 255, 0),    # 正常 - 绿色
        (0, 165, 255),  # 气孔 - 橙色
        (0, 0, 255),    # 裂纹 - 红色
        (255, 0, 0)     # 夹渣 - 蓝色
    ]
    
    # 绘制检测框
    for res in results:
        x1, y1, x2, y2 = map(int, res[:4])
        cnn_cls = int(res[6])
        color = colors[cnn_cls % len(colors)]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
    
    # OpenCV BGR转RGB用于matplotlib显示
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    plt.imshow(vis_image)
    plt.title('检测结果')
    plt.axis('off')
    
    # 图2：饼图显示缺陷类型分布
    plt.subplot(2, 2, 2)
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('缺陷类型分布')
    
    # 图3：柱状图显示缺陷数量
    plt.subplot(2, 2, 3)
    bars = plt.bar(labels, sizes, color=['green', 'orange', 'red', 'blue'])
    plt.title('缺陷数量统计')
    plt.xlabel('缺陷类型')
    plt.ylabel('数量')
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    # 图4：置信度分布
    plt.subplot(2, 2, 4)
    confidences = [res[7] for res in results]  # CNN置信度
    class_ids = [int(res[6]) for res in results]  # CNN类别ID
    
    # 创建散点图，每种类别用不同颜色
    for cls_id in set(class_ids):
        if cls_id < len(class_names):
            cls_name = class_names[cls_id]
            cls_confs = [confidences[i] for i in range(len(confidences)) if class_ids[i] == cls_id]
            cls_indices = [i for i in range(len(class_ids)) if class_ids[i] == cls_id]
            plt.scatter(cls_indices, cls_confs, label=cls_name)
    
    plt.title('检测置信度分布')
    plt.xlabel('缺陷索引')
    plt.ylabel('置信度')
    plt.legend()
    plt.ylim(0, 1.1)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"可视化结果已保存到 {output_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误: 无法读取图像 {args.image}")
        return
    
    # 获取检测结果
    if args.result and os.path.exists(args.result):
        # 从文件加载结果
        results = load_results(args.result)
        print(f"从文件 {args.result} 加载了 {len(results)} 个检测结果")
    else:
        # 执行检测
        if not os.path.exists(args.yolo_model) or not os.path.exists(args.cnn_model):
            print(f"错误: 模型文件不存在")
            return
        
        print("初始化级联检测器...")
        detector = CascadeDetector(
            yolo_model_path=args.yolo_model,
            cnn_model_path=args.cnn_model,
            num_classes=args.num_classes,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres
        )
        
        print("执行检测...")
        results = detector.detect(image)
        print(f"检测到 {len(results)} 个缺陷")
    
    # 准备输出路径
    basename = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join(args.output_dir, f"{basename}_analysis.png")
    
    # 可视化结果
    print("生成可视化分析...")
    visualize_defect_types(image, results, output_path)
    
    print("可视化分析完成！")


if __name__ == '__main__':
    main() 