#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

def main():
    """运行数据集转换脚本"""
    print("=" * 50)
    print("焊缝缺陷检测系统 - 数据集处理工具")
    print("=" * 50)
    
    # 检查YOLO数据集路径
    yolo_path = 'data/welding_defects/yolov8'
    if not os.path.exists(yolo_path):
        print(f"错误：YOLO数据集路径不存在: {yolo_path}")
        return 1
    
    # 检查YAML配置文件
    yaml_path = os.path.join(yolo_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"错误：YAML配置文件不存在: {yaml_path}")
        return 1
    
    # 设置输出路径
    output_path = 'data/welding_defects/cnn'
    os.makedirs(output_path, exist_ok=True)
    
    # 执行数据集分割
    print("\n1. 分割YOLO数据集（30%验证集，10%测试集）...")
    try:
        cmd = [
            sys.executable, 'convert_dataset.py',
            '--yolo-path', yolo_path,
            '--output-path', output_path,
            '--split-yolo',
            '--train-ratio', '0.6',
            '--val-ratio', '0.3',
            '--test-ratio', '0.1'
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"错误：执行转换脚本时出错: {e}")
        return 1
    
    print("\n数据集处理完成！")
    print(f"CNN数据集保存在: {output_path}")
    print("=" * 50)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 