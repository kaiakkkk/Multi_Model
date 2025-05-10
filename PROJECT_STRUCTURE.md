# 焊缝缺陷多模型级联检测系统 - 项目结构

本文档详细说明项目的文件和目录结构，帮助您理解系统组织。

## 核心目录

### 1. models/
模型定义文件：
- `cascade_detector.py` - 级联检测器定义
- `cnn_model.py` - CNN模型架构
- `yolo_model.py` - YOLO模型封装

### 2. data/
数据集和数据处理文件：
- `data/welding_defects/yolov8/` - YOLO格式的原始数据集，包含标签文件和图像
- `data/welding_defects/cnn/` - 转换后的CNN训练数据，按类别组织
  - `train/` - 训练集
  - `val/` - 验证集
  - `test/` - 测试集

### 3. utils/
工具函数和辅助模块，按功能分类：

#### 3.1 utils/data_tools/
数据处理相关工具：
- `convert_yolo_to_cnn.py` - YOLO格式转CNN格式的核心逻辑
- `data_utils.py` - 数据处理相关工具函数
- `convert_dataset.py` - 数据集转换命令行工具
- `check_dataset.py` - 数据集检查和可视化
- `run_conversion_light.py` - 简化的转换执行脚本

#### 3.2 utils/test_tools/
测试相关工具：
- `test_yolo.py` - YOLO模型单独测试
- `test_cascade_model_light.py` - 完整级联模型轻量化测试
- `test_cascade_simple_light.py` - 简化级联系统轻量化测试

#### 3.3 utils/demo_tools/
演示相关工具：
- `run_demo.py` - 运行系统演示的脚本
- `demo.py` - 演示应用程序

### 4. results/
模型训练和推理结果：
- 模型权重
- 训练日志
- 性能评估

### 5. results_light/
轻量级测试和样例数据：
- `cascade_simple/` - 简化级联系统测试结果
- `test_cascade/` - 级联测试结果
- `yolo_test/` - YOLO测试结果
- `dataset_check/` - 数据集检查结果

## 核心脚本

### 主程序
- `run_all.py` - 统一的命令行接口，可以调用系统的各个功能模块

### 核心功能脚本
- `train.py` - 模型训练主脚本
- `detect.py` - 缺陷检测脚本
- `visualize.py` - 结果可视化和分析

## 命令行使用说明

系统提供统一的命令行接口 `run_all.py`，通过子命令调用不同功能：

### 数据处理
```bash
# 转换数据集
python run_all.py data convert --yolo-dir data/welding_defects/yolov8 --output-dir data/welding_defects/cnn

# 检查数据集
python run_all.py data check
```

### 模型训练
```bash
# 训练完整级联系统
python run_all.py model train --data-dir data/welding_defects

# 仅训练YOLO模型
python run_all.py model train --data-dir data/welding_defects --yolo-only

# 仅训练CNN模型
python run_all.py model train --data-dir data/welding_defects --cnn-only
```

### 缺陷检测
```bash
# 检测图像
python run_all.py detect --source path/to/image.jpg

# 检测视频并显示结果
python run_all.py detect --source path/to/video.mp4 --view-img
```

### 结果可视化
```bash
python run_all.py visualize --image path/to/image.jpg
```

### 测试功能
```bash
# 测试YOLO模型
python run_all.py test yolo

# 测试级联系统
python run_all.py test cascade
```

### 运行演示
```bash
# 运行所有演示
python run_all.py demo --all

# 运行特定演示
python run_all.py demo --data
python run_all.py demo --yolo
python run_all.py demo --cascade
python run_all.py demo --detect
``` 