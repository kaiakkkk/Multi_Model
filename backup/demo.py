#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统演示脚本
"""

import os
import argparse
import cv2
import torch
from pathlib import Path
from models.cascade_detector import CascadeDetector
from utils.data_utils import download_sample_dataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测系统演示脚本')
    
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'image', 'video'],
                        help='演示模式: webcam(摄像头), image(图像), video(视频)')
    parser.add_argument('--source', type=str, default=None,
                        help='输入源，在image或video模式下需要指定')
    parser.add_argument('--setup-data', action='store_true',
                        help='设置示例数据集目录结构')
    parser.add_argument('--output-dir', type=str, default='runs/demo',
                        help='输出目录')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='使用预训练模型（不需要自己训练的模型）')
    
    return parser.parse_args()


def setup_demo_environment():
    """设置演示环境"""
    print("=" * 50)
    print("设置演示环境")
    print("=" * 50)
    
    # 检查依赖库
    try:
        import ultralytics
        import torch
        import torchvision
        print("✓ 所有依赖库已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖库: {e}")
        print("请运行 'pip install -r requirements.txt' 安装所有依赖")
        return False
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        print(f"✓ CUDA可用 (版本: {cuda_version})")
    else:
        print("! CUDA不可用，将使用CPU进行演示 (速度可能较慢)")
    
    # 检查摄像头
    if args.mode == 'webcam':
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ 摄像头工作正常")
                h, w = frame.shape[:2]
                print(f"  摄像头分辨率: {w}x{h}")
            else:
                print("✗ 无法从摄像头读取图像")
                return False
            cap.release()
        else:
            print("✗ 无法访问摄像头")
            return False
    
    return True


def run_demo_webcam(detector, output_dir):
    """运行摄像头演示"""
    print("=" * 50)
    print("启动摄像头演示模式")
    print("按'q'键退出")
    print("=" * 50)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 创建输出视频
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join(output_dir, "webcam_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测
        results = detector.detect(frame)
        
        # 可视化结果
        vis_frame = detector.visualize(frame, results)
        
        # 添加系统信息
        info_text = "YOLOv8 + CNN级联检测系统"
        cv2.putText(vis_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        fps_text = f"检测到 {len(results)} 个缺陷"
        cv2.putText(vis_frame, fps_text, (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow('YOLOv8 + CNN级联检测', vis_frame)
        
        # 写入输出视频
        writer.write(vis_frame)
        
        # 按q键退出
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"演示视频已保存到 {output_video_path}")


def run_demo_image(detector, image_path, output_dir):
    """运行图像演示"""
    print("=" * 50)
    print(f"对图像 {image_path} 进行检测")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
        return
    
    # 执行检测
    results = detector.detect(image)
    
    # 可视化结果
    vis_image = detector.visualize(image, results)
    
    # 添加系统信息
    h, w = vis_image.shape[:2]
    info_text = "YOLOv8 + CNN级联检测系统"
    cv2.putText(vis_image, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    result_text = f"检测到 {len(results)} 个缺陷"
    cv2.putText(vis_image, result_text, (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 保存结果
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_result.jpg")
    cv2.imwrite(output_path, vis_image)
    
    # 显示结果
    cv2.imshow('检测结果', vis_image)
    print("按任意键关闭窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"结果已保存到 {output_path}")


def run_demo_video(detector, video_path, output_dir):
    """运行视频演示"""
    print("=" * 50)
    print(f"对视频 {video_path} 进行检测")
    print("按'q'键退出")
    print("=" * 50)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔5帧处理一次（提高处理速度）
        if frame_idx % 5 == 0:
            # 执行检测
            results = detector.detect(frame)
            
            # 可视化结果
            vis_frame = detector.visualize(frame, results)
            
            # 添加系统信息
            info_text = "YOLOv8 + CNN级联检测系统"
            cv2.putText(vis_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            fps_text = f"帧: {frame_idx} | 检测到: {len(results)}个缺陷"
            cv2.putText(vis_frame, fps_text, (10, height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            vis_frame = frame
        
        # 写入输出视频
        writer.write(vis_frame)
        
        # 显示结果
        cv2.imshow('检测结果', vis_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"处理帧: {frame_idx}")
    
    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"结果视频已保存到 {output_path}")


def main():
    """主函数"""
    global args
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置示例数据集
    if args.setup_data:
        download_sample_dataset('data/welding_defects')
    
    # 设置演示环境
    if not setup_demo_environment():
        print("环境设置失败，请检查上述错误并修复")
        return
    
    # 初始化检测器
    print("初始化级联检测器...")
    try:
        if args.use_pretrained:
            # 使用预训练模型
            detector = CascadeDetector()
        else:
            # 检查本地训练模型
            yolo_model = 'runs/train/yolo_model.pt'
            cnn_model = 'runs/train/cnn_model.pt'
            
            if not os.path.exists(yolo_model) or not os.path.exists(cnn_model):
                print("本地训练模型不存在，将使用默认模型")
                detector = CascadeDetector()
            else:
                detector = CascadeDetector(
                    yolo_model_path=yolo_model,
                    cnn_model_path=cnn_model
                )
        
        print("级联检测器初始化完成")
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        return
    
    # 运行演示
    if args.mode == 'webcam':
        run_demo_webcam(detector, args.output_dir)
    elif args.mode == 'image':
        if args.source is None:
            print("错误: 图像模式需要指定--source参数")
            return
        run_demo_image(detector, args.source, args.output_dir)
    elif args.mode == 'video':
        if args.source is None:
            print("错误: 视频模式需要指定--source参数")
            return
        run_demo_video(detector, args.source, args.output_dir)
    
    print("演示完成!")


if __name__ == '__main__':
    main() 