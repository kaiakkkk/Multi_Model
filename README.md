# 焊缝缺陷多模型级联检测系统

本系统采用YOLOv8与CNN级联架构，用于焊缝缺陷的高精度检测。

## 系统架构
- **第一阶段**：YOLOv8快速定位焊缝区域中的潜在缺陷候选框
- **第二阶段**：CNN对YOLOv8输出的候选区域进行高分辨率特征提取，提升对气孔、裂纹等细微缺陷的分类精度

## 支持的缺陷类型
系统支持五种焊缝缺陷检测类型：
- 'CC': 焊接裂缝（Cold Cracking）
- 'NG': 未融合（Not Good Fusion）
- 'PX': 气孔缺陷（Porosity Defect）
- 'QP': 气泡缺陷（Gas Bubble）
- 'SX': 水纹缺陷（Water Pattern）

## 环境配置
系统要求：
- Python 3.8+
- CUDA 11.0+（推荐，用于GPU加速）
- 内存 8GB+

安装依赖：
```bash
conda activate yolov8_new
pip install -r requirements.txt
```

## 统一命令行接口

系统提供了统一的命令行接口`run_all.py`，可用于调用各项功能：

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

## 原始命令（仍然可用）

### 数据处理流程
1. 将YOLOv8格式的标注数据转换为CNN训练所需的格式
```bash
python utils/data_tools/convert_dataset.py --yolo-dir data/welding_defects/yolov8 --output-dir data/welding_defects/cnn
```

2. 检查数据集转换结果
```bash
python utils/data_tools/check_dataset.py
```

### 模型训练
```bash
python train.py --data-dir data/welding_defects --batch-size 16 --yolo-epochs 100 --cnn-epochs 50
```

### 缺陷检测
```bash
python detect.py --source path/to/image.jpg
```

## 性能优化建议
1. **数据均衡**：CC和NG类别样本严重不足，建议进行数据增强
2. **模型微调**：对YOLOv8模型进行特定任务微调
3. **置信度阈值**：调整置信度阈值以平衡检测性能

## 文件结构
详细的文件结构说明请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。 