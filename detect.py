#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统检测脚本
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from models.cascade_detector import CascadeDetector
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测系统检测脚本')
    
    # 输入参数
    parser.add_argument('--source', type=str, required=True,
                        help='输入图像或视频的路径，或者0表示摄像头')
    
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
    parser.add_argument('--output-dir', type=str, default='runs/detect',
                        help='输出目录')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存检测结果为文本文件')
    parser.add_argument('--view-img', action='store_true',
                        help='显示检测结果')
    
    # 性能参数
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像大小')
    
    return parser.parse_args()


def detect_image(detector, image_path, output_dir, view_img=False, save_txt=False):
    """
    检测单张图像
    
    参数:
        detector: 级联检测器
        image_path: 图像路径
        output_dir: 输出目录
        view_img: 是否显示图像
        save_txt: 是否保存结果为文本文件
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
        return
    
    # 执行检测
    results = detector.detect(image)
    
    # 可视化结果
    vis_image = detector.visualize(image, results)
    
    # 准备输出路径
    filename = os.path.basename(image_path)
    basename = os.path.splitext(filename)[0]
    
    # 保存可视化图像
    output_img_path = os.path.join(output_dir, f"{basename}_result.jpg")
    cv2.imwrite(output_img_path, vis_image)
    print(f"结果已保存到 {output_img_path}")
    
    # 保存文本结果
    if save_txt:
        output_txt_path = os.path.join(output_dir, f"{basename}_result.txt")
        with open(output_txt_path, 'w') as f:
            for res in results:
                x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                # 格式: <x1> <y1> <x2> <y2> <yolo_conf> <yolo_class_id> <cnn_class_id> <cnn_conf>
                f.write(f"{x1} {y1} {x2} {y2} {yolo_conf:.4f} {int(yolo_cls)} {int(cnn_cls)} {cnn_conf:.4f}\n")
        print(f"检测结果已保存到 {output_txt_path}")
    
    # 显示结果
    if view_img:
        cv2.imshow('Detection Result', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def detect_video(detector, video_path, output_dir, view_img=False):
    """
    检测视频
    
    参数:
        detector: 级联检测器
        video_path: 视频路径
        output_dir: 输出目录
        view_img: 是否显示视频
    """
    # 打开视频
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        video_name = 'camera'
    else:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if not cap.isOpened():
        print(f"无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频写入器
    output_video_path = os.path.join(output_dir, f"{video_name}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测
        results = detector.detect(frame)
        
        # 可视化结果
        vis_frame = detector.visualize(frame, results)
        
        # 写入输出视频
        writer.write(vis_frame)
        
        # 显示结果
        if view_img:
            cv2.imshow('Detection Result', vis_frame)
            if cv2.waitKey(1) == ord('q'):  # 按q键退出
                break
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"处理帧 {frame_idx}")
    
    # 释放资源
    cap.release()
    writer.release()
    if view_img:
        cv2.destroyAllWindows()
    
    print(f"结果视频已保存到 {output_video_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.yolo_model):
        print(f"错误: YOLOv8模型文件 {args.yolo_model} 不存在")
        return
    
    if not os.path.exists(args.cnn_model):
        print(f"错误: CNN模型文件 {args.cnn_model} 不存在")
        return
    
    # 初始化级联检测器
    print("初始化级联检测器...")
    detector = CascadeDetector(
        yolo_model_path=args.yolo_model,
        cnn_model_path=args.cnn_model,
        num_classes=args.num_classes,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        cnn_input_size=(224, 224)  # 固定CNN输入大小
    )
    
    # 判断输入是图像还是视频
    source = args.source
    is_video = False
    
    if source.isdigit():  # 摄像头
        is_video = True
    elif os.path.isfile(source):
        # 判断文件类型
        ext = os.path.splitext(source)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    else:
        print(f"错误: 输入源 {source} 不存在")
        return
    
    # 执行检测
    print(f"开始检测 {'视频' if is_video else '图像'}...")
    
    if is_video:
        detect_video(detector, source, args.output_dir, args.view_img)
    else:
        detect_image(detector, source, args.output_dir, args.view_img, args.save_txt)
    
    print("检测完成！")


if __name__ == '__main__':
    main() 