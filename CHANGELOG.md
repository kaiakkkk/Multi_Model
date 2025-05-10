# 变更日志

## 2025-05-10 项目结构重构

### 添加
- 创建了统一的命令行接口 `run_all.py`，提供所有功能的访问入口
- 在utils目录下创建了三个子目录，按功能分类组织工具：
  - `utils/data_tools/` - 数据处理工具
  - `utils/test_tools/` - 测试工具
  - `utils/demo_tools/` - 演示工具
- 为各个包添加了 `__init__.py` 文件，使其可以正确导入

### 修改
- 移动了工具类文件到对应的utils子目录：
  - 数据处理相关工具移至 `utils/data_tools/`
  - 测试相关工具移至 `utils/test_tools/`
  - 演示相关工具移至 `utils/demo_tools/`
- 更新了 `README.md` 和 `PROJECT_STRUCTURE.md`，反映新的项目结构
- 保留了原始脚本的单独使用方式，同时提供了新的统一命令行接口

### 优化
- 简化了根目录，只保留核心脚本和配置文件
- 提供了更加模块化和可扩展的项目结构
- 改进了命令行接口，使用子命令组织不同功能

## 2025-05-10 项目整理

### 添加
- 创建了综合性README.md文件，合并了原README.md和USAGE.md的内容
- 添加了PROJECT_STRUCTURE.md文件，详细说明项目结构
- 添加了run_demo.py演示脚本，方便快速测试系统
- 在requirements.txt中添加了额外依赖：tensorboard、seaborn和imgaug
- 创建了CHANGELOG.md跟踪项目变更

### 修改
- 将轻量级测试相关文件重命名为*_light：
  - test_cascade_simple.py → test_cascade_simple_light.py
  - test_cascade_model.py → test_cascade_model_light.py
  - run_conversion.py → run_conversion_light.py
- 将测试相关结果目录移至results_light/目录下：
  - cascade_simple/
  - test_cascade/
  - yolo_test/
  - dataset_check/

### 删除
- 删除了冗余的USAGE.md文件，内容已合并到README.md

### 结构优化
- 明确区分了正式训练数据和轻量级测试数据
- 优化了文档结构，提高了项目可读性
- 整理了项目文件，使结构更加清晰 