#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统 - 主程序
提供统一的命令行接口调用各功能模块
"""

import os
import sys
import argparse
import logging

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="焊缝缺陷多模型级联检测系统")
    subparsers = parser.add_subparsers(dest="command", help="选择操作命令")
    
    # 数据处理命令
    data_parser = subparsers.add_parser("data", help="数据处理相关命令")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    
    # 数据转换
    convert_parser = data_subparsers.add_parser("convert", help="转换数据集")
    convert_parser.add_argument("--yolo-dir", required=True, help="YOLO数据集目录")
    convert_parser.add_argument("--output-dir", required=True, help="输出目录")
    
    # 数据检查
    check_parser = data_subparsers.add_parser("check", help="检查数据集")
    
    # 模型命令
    model_parser = subparsers.add_parser("model", help="模型相关命令")
    model_subparsers = model_parser.add_subparsers(dest="model_command")
    
    # 训练模型
    train_parser = model_subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data-dir", required=True, help="数据集目录")
    train_parser.add_argument("--yolo-only", action="store_true", help="仅训练YOLO模型")
    train_parser.add_argument("--cnn-only", action="store_true", help="仅训练CNN模型")
    
    # 检测命令
    detect_parser = subparsers.add_parser("detect", help="检测焊缝缺陷")
    detect_parser.add_argument("--source", required=True, help="输入图像或视频")
    detect_parser.add_argument("--view-img", action="store_true", help="显示检测结果")
    
    # 可视化命令
    vis_parser = subparsers.add_parser("visualize", help="可视化检测结果")
    vis_parser.add_argument("--image", required=True, help="输入图像")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试系统功能")
    test_subparsers = test_parser.add_subparsers(dest="test_command")
    
    # YOLO测试
    yolo_test_parser = test_subparsers.add_parser("yolo", help="测试YOLO模型")
    
    # 级联系统测试
    cascade_test_parser = test_subparsers.add_parser("cascade", help="测试级联系统")
    
    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="运行系统演示")
    demo_parser.add_argument("--all", action="store_true", help="运行所有演示")
    demo_parser.add_argument("--data", action="store_true", help="数据处理演示")
    demo_parser.add_argument("--yolo", action="store_true", help="YOLO测试演示")
    demo_parser.add_argument("--cascade", action="store_true", help="级联系统演示")
    demo_parser.add_argument("--detect", action="store_true", help="缺陷检测演示")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # 处理数据命令
    if args.command == "data":
        if args.data_command == "convert":
            from data_tools.convert_dataset import main as convert_main
            sys.argv = [sys.argv[0]] + [f"--yolo-dir={args.yolo_dir}", f"--output-dir={args.output_dir}"]
            return convert_main()
        elif args.data_command == "check":
            from data_tools.check_dataset import main as check_main
            sys.argv = [sys.argv[0], "--script-path=utils/data_tools/check_dataset.py"]
            return check_main()
            
    # 处理模型命令
    elif args.command == "model":
        if args.model_command == "train":
            # 导入训练模块
            from train import main as train_main
            
            # 重构命令行参数
            new_args = [sys.argv[0], f"--data-dir={args.data_dir}"]
            if args.yolo_only:
                new_args.append("--yolo-only")
            if args.cnn_only:
                new_args.append("--cnn-only")
                
            sys.argv = new_args
            return train_main()
            
    # 处理检测命令
    elif args.command == "detect":
        # 导入检测模块
        from detect import main as detect_main
        
        # 重构命令行参数
        new_args = [sys.argv[0], f"--source={args.source}"]
        if args.view_img:
            new_args.append("--view-img")
            
        sys.argv = new_args
        return detect_main()
        
    # 处理可视化命令
    elif args.command == "visualize":
        # 导入可视化模块
        from visualize import main as visualize_main
        
        # 重构命令行参数
        sys.argv = [sys.argv[0], f"--image={args.image}"]
        return visualize_main()
        
    # 处理测试命令
    elif args.command == "test":
        if args.test_command == "yolo":
            from utils.demo_tools.run_demo import test_yolo as test_yolo_main
            sys.argv = [sys.argv[0], "--yolo"]
            return test_yolo_main()
        elif args.test_command == "cascade":
            from utils.demo_tools.run_demo import test_cascade as test_cascade_main
            sys.argv = [sys.argv[0], "--cascade"]
            return test_cascade_main()
            
    # 处理演示命令
    elif args.command == "demo":
        from demo_tools.run_demo import main as demo_main
        
        # 重构命令行参数
        new_args = [sys.argv[0]]
        if args.all:
            new_args.append("--all")
        if args.data:
            new_args.append("--data")
        if args.yolo:
            new_args.append("--yolo")
        if args.cascade:
            new_args.append("--cascade")
        if args.detect:
            new_args.append("--detect")
            
        sys.argv = new_args
        return demo_main()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 