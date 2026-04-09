# 实验管理系统使用指南

## 概述

这个实验管理系统可以帮助你系统地管理参数实验，自动记录每次运行的参数和结果，方便后续分析对比。

## 文件结构

```
JS24/
├── experiment_tracker.py      # 实验跟踪核心模块
├── experiment_config.yaml     # 实验配置文件
├── run_experiments.py         # 实验运行脚本
├── 06_Experiment_Analysis.ipynb  # 结果分析notebook
└── experiments/               # 实验数据目录（自动创建）
    ├── experiment_history.json  # 实验历史记录
    └── experiments.csv          # CSV格式的实验结果
```

## 快速开始

### 1. 查看可用配置

```bash
cd janeStreet/JS24
python run_experiments.py --list
```

### 2. 运行单个实验

```bash
# 运行标准配置
python run_experiments.py --experiment standard

# 运行快速测试配置
python run_experiments.py --experiment quick_test

# 运行高质量配置
python run_experiments.py --experiment high_quality
```

### 3. 运行所有预设实验

```bash
python run_experiments.py --all
```

### 4. 运行网格搜索

```bash
# 搜索学习率和最大深度的最佳组合
python run_experiments.py --grid_search --param learning_rate,max_depth

# 搜索多个参数
python run_experiments.py --grid_search --param learning_rate,max_depth,num_leaves
```

### 5. 分析实验结果

运行实验后，打开 `06_Experiment_Analysis.ipynb` notebook查看结果可视化。

## 配置文件说明

`experiment_config.yaml` 包含以下部分：

### 基础配置 (base_config)

所有实验共享的基础参数：

```yaml
base_config:
  data:
    n_dates_to_load: 15      # 加载的天数
    n_symbols_to_use: 20     # 使用的股票数
  training:
    seed: 42
    n_folds: 3
    early_stopping_rounds: 50
```

### 网格搜索配置 (grid_search)

定义参数搜索空间：

```yaml
grid_search:
  lgbm_params:
    learning_rate: [0.01, 0.03, 0.05, 0.1]
    max_depth: [4, 6, 8]
    num_leaves: [32, 64, 128]
    # ... 更多参数
```

### 预设实验 (experiments)

预定义的参数组合：

```yaml
experiments:
  standard:
    name: "标准配置"
    learning_rate: 0.05
    max_depth: 6
    # ... 更多参数
```

## 自定义实验

### 方法1: 修改配置文件

编辑 `experiment_config.yaml`，在 `experiments` 部分添加新配置：

```yaml
experiments:
  my_custom_experiment:
    name: "我的自定义实验"
    n_dates_to_load: 25
    n_symbols_to_use: 30
    learning_rate: 0.02
    max_depth: 7
    num_leaves: 96
    n_estimators: 800
```

然后运行：

```bash
python run_experiments.py --experiment my_custom_experiment
```

### 方法2: 修改网格搜索配置

编辑 `experiment_config.yaml` 中的 `grid_search` 部分：

```yaml
grid_search:
  lgbm_params:
    learning_rate: [0.01, 0.02, 0.05, 0.1]  # 修改候选值
    max_depth: [5, 6, 7, 8]                # 添加更多候选值
```

### 方法3: 使用Python代码

在notebook或Python脚本中使用实验跟踪器：

```python
from experiment_tracker import ExperimentTracker
from run_experiments import ExperimentRunner

# 创建运行器
runner = ExperimentRunner()

# 运行实验
metrics = runner.run_single_experiment('standard')

print(f"验证集R²: {metrics['valid_r2']:.6f}")
```

## 实验结果分析

### 查看实验历史

实验历史保存在：
- `experiments/experiment_history.json` - JSON格式
- `experiments/experiments.csv` - CSV格式（可用Excel打开）

### 在Jupyter Notebook中分析

1. 打开 `06_Experiment_Analysis.ipynb`
2. 运行所有单元格
3. 查看自动生成的可视化：
   - 参数影响分析
   - 参数交互效应
   - 过拟合分析
   - 实验时间线
   - 最佳实验对比

### 常用分析命令

```python
from experiment_tracker import ExperimentTracker

# 创建跟踪器
tracker = ExperimentTracker()

# 查看摘要
tracker.print_summary()

# 获取最佳实验
best = tracker.get_best_experiments('valid_r2', n=5)

# 对比特定实验
comparison = tracker.compare_experiments(['exp_id_1', 'exp_id_2'])

# 保存为CSV
tracker.save_to_csv()
```

## 典型工作流程

### 场景1: 快速迭代测试

```bash
# 1. 使用快速测试配置
python run_experiments.py --experiment quick_test

# 2. 查看结果
python 06_Experiment_Analysis.ipynb  # 在Jupyter中打开
```

### 场景2: 系统性参数搜索

```bash
# 1. 运行网格搜索
python run_experiments.py --grid_search --param learning_rate,max_depth

# 2. 分析结果找出最佳参数范围
# 打开 06_Experiment_Analysis.ipynb

# 3. 基于分析结果细化搜索
# 编辑 experiment_config.yaml，调整参数范围

# 4. 重新运行网格搜索
python run_experiments.py --grid_search --param num_leaves,min_data_in_leaf
```

### 场景3: 对比不同配置

```bash
# 1. 运行多个预设配置
python run_experiments.py --all

# 2. 在notebook中对比
# 打开 06_Experiment_Analysis.ipynb
```

## 高级用法

### 自定义评估指标

编辑 `run_experiments.py` 中的 `run_single_experiment` 方法，添加自定义指标：

```python
# 添加自定义指标
metrics = {
    'train_r2': train_r2,
    'valid_r2': valid_r2,
    'custom_metric': your_custom_calculation()
}
```

### 并行运行实验

使用GNU parallel或类似工具：

```bash
# 并行运行多个实验
parallel python run_experiments.py --experiment {} ::: standard quick_test high_quality
```

### 导出最佳模型

最佳模型会自动保存在 `models/` 目录下，文件名格式为 `model_<experiment_name>.pkl`。

## 常见问题

**Q: 实验历史文件太大怎么办？**

A: 可以删除旧的实验记录或创建新的实验目录：
```python
tracker = ExperimentTracker(experiment_dir="./experiments_v2")
```

**Q: 如何恢复之前的实验记录？**

A: 实验记录保存在 `experiments/experiment_history.json`，确保不要删除这个文件。

**Q: 网格搜索时间太长？**

A:
1. 减少参数候选值数量
2. 使用快速测试配置验证流程
3. 考虑使用更智能的搜索方法（如贝叶斯优化）

**Q: 如何分享实验结果？**

A:
1. 使用 `tracker.save_to_csv()` 导出CSV
2. 在notebook中生成可视化图表
3. 分享 `experiments/` 目录

## 下一步

1. 运行第一个实验熟悉流程
2. 根据需要修改配置文件
3. 使用分析notebook理解参数影响
4. 迭代优化找到最佳配置
