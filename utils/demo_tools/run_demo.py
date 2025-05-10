#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统 - 演示脚本
快速运行系统功能的演示示例
"""

import os
import sys
import argparse
import logging
import subprocess
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, desc=None):
    """运行命令并打印输出"""
    if desc:
        logger.info(f"[{desc}] 运行命令: {cmd}")
    else:
        logger.info(f"运行命令: {cmd}")
        
    start_time = time.time()
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line.strip())
        
    process.wait()
    end_time = time.time()
    
    if process.returncode == 0:
        logger.info(f"命令成功执行，耗时 {end_time - start_time:.2f} 秒")
    else:
        logger.error(f"命令执行失败，返回代码 {process.returncode}")
        
    return process.returncode

def demo_data_conversion():
    """演示数据转换流程"""
    logger.info("=== 数据转换演示 ===")
    
    # 运行数据转换
    run_command(
        "python convert_dataset.py --yolo-dir data/welding_defects/yolov8 --output-dir data/welding_defects/cnn",
        "数据转换"
    )
    
    # 检查数据集
    run_command("python check_dataset.py", "数据集检查")

def demo_test_yolo():
    """演示YOLO测试"""
    logger.info("=== YOLO测试演示 ===")
    
    # 测试YOLO模型
    run_command("python test_yolo.py", "YOLO测试")

def demo_test_cascade():
    """演示级联系统测试"""
    logger.info("=== 级联系统测试演示 ===")
    
    # 测试轻量化级联系统
    run_command("python test_cascade_simple_light.py", "轻量级级联测试")

def demo_detection():
    """演示缺陷检测"""
    logger.info("=== 缺陷检测演示 ===")
    
    # 获取测试图像
    test_img = None
    test_dir = "results_light/yolo_test/test_images"
    
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
        if test_files:
            test_img = os.path.join(test_dir, test_files[0])
    
    if not test_img:
        logger.warning("未找到测试图像，跳过检测演示")
        return
        
    # 运行检测
    run_command(f"python detect.py --source {test_img}", "缺陷检测")
    
    # 可视化结果
    run_command(f"python visualize.py --image {test_img}", "结果可视化")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="焊缝缺陷检测系统演示")
    parser.add_argument("--all", action="store_true", help="运行所有演示")
    parser.add_argument("--data", action="store_true", help="演示数据转换")
    parser.add_argument("--yolo", action="store_true", help="演示YOLO测试")
    parser.add_argument("--cascade", action="store_true", help="演示级联系统测试")
    parser.add_argument("--detect", action="store_true", help="演示缺陷检测")
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认运行所有演示
    run_all = args.all or (not args.data and not args.yolo and not args.cascade and not args.detect)
    
    try:
        if run_all or args.data:
            demo_data_conversion()
            
        if run_all or args.yolo:
            demo_test_yolo()
            
        if run_all or args.cascade:
            demo_test_cascade()
            
        if run_all or args.detect:
            demo_detection()
            
        logger.info("演示完成！")
        
    except Exception as e:
        logger.exception(f"演示过程中发生错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 