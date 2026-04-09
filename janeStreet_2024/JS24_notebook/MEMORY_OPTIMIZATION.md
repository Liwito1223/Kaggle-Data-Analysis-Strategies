# Jane Street 项目 - 内存优化说明

## 📊 当前问题分析

### 问题1: Validation比Training数据多

**现象**:
- Training: 15.9M行 (3.9GB)
- Validation: 23.7M行 (5.8GB)

**原因**:
滞后特征工程代码有bug，validation包含了不应该包含的数据：
1. Validation包含了date_id 500-509的数据（应该被skip）
2. Validation包含了date_id 1599-1698的数据（更新的数据）

**解决方案**:
1. 重新运行滞后特征工程，修正分割逻辑
2. 或者在训练时通过采样来平衡数据

### 问题2: Mac内存超限

**原因**:
- 3.9GB的parquet文件转换成pandas需要**15-20GB内存**
- 一次性加载整个文件导致内存溢出

**内存使用估算**:
| 操作 | 内存占用 |
|------|---------|
| 读取3.9GB parquet | ~4GB |
| 转换为pandas | +10-15GB |
| **总计** | **~15-20GB** |

## ✅ 解决方案

我已经创建了内存安全的加载函数：

### 1. utils.py - 内存优化工具

提供了两个函数：

#### `load_data_memory_safe()` - 智能内存管理
- 自动检测可用内存
- 根据内存限制自动采样
- 使用polars lazy loading
- 保持时间序列连续性

#### `load_data_minimal()` - 快速测试
- 只加载指定数量的样本
- 适合快速测试代码

### 2. 更新的训练笔记本

更新了 `03_Training_LightGBM.ipynb` 的数据加载部分：
- 优先使用utils中的内存安全函数
- 备用基础方法（保守采样100天训练+30天验证）
- 实时内存监控

## 🚀 使用方法

### 方式1: 使用utils模块（推荐）

```python
from utils import load_data_memory_safe

train_df, valid_df = load_data_memory_safe(
    train_file='./processed_data/training.parquet',
    valid_file='./processed_data/validation.parquet',
    feature_cols=FEATURE_COLS,
    max_memory_gb=6,  # 设置保守的内存限制
    use_lazy=True
)
```

### 方式2: 使用更新后的笔记本

直接运行 `03_Training_LightGBM.ipynb`，它会自动：
1. 尝试导入utils模块
2. 如果失败，使用内置的基础方法
3. 自动采样避免内存溢出

## 📋 采样策略说明

### 训练数据
- 默认保留最后**100天**的数据
- 约300-400万行（vs 原始1590万行）
- 内存占用约3-4GB

### 验证数据
- 默认保留最后**30天**的数据
- 约100万行（vs 原始2370万行）
- 内存占用约1-2GB

### 为什么这样采样？
1. **保留最新的数据**: 金融数据最近的数据更具代表性
2. **保持时间连续性**: 按日期采样而不是随机采样
3. **平衡训练效果**: 100天训练+30天验证足够训练出好模型

## 🔧 自定义采样比例

如果内存更充足或更紧张，可以调整采样：

```python
# 在utils.py中修改
train_df, valid_df = load_data_memory_safe(
    ...
    sample_ratio=0.5,  # 使用50%的数据
    max_memory_gb=8,   # 如果有8GB可用内存
    ...
)
```

或在基础方法中修改：
```python
# 保留更多天数的训练数据
selected_dates = all_dates[-200:]  # 改为200天
```

## ⚠️ 重要提示

1. **重新生成训练数据**: 建议修复滞后特征工程的bug后重新生成数据
2. **监控内存**: 训练时打开Activity Monitor监控内存使用
3. **逐步测试**: 先用小数据测试代码，再逐步增加数据量
4. **保存模型**: 训练好的模型保存到`models/`文件夹

## 📈 预期性能

使用采样数据训练的模型：
- 训练时间: ~5-10分钟（vs 全数据数小时）
- 内存占用: <6GB（vs 全数据15-20GB）
- 模型性能: 仍然可以达到合理的R²得分

## 🎯 下一步

1. **运行更新后的训练笔记本**
2. **监控内存使用情况**
3. **如果成功，可以尝试增加采样比例**
4. **对比不同采样比例的模型效果**
