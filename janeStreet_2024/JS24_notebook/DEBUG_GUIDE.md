# Kernel崩溃调试指南

## 目录
1. [快速诊断](#快速诊断)
2. [内存监控工具使用](#内存监控工具使用)
3. [常见问题诊断](#常见问题诊断)
4. [数据加载优化](#数据加载优化)
5. [模型训练优化](#模型训练优化)
6. [调试清单](#调试清单)

---

## 快速诊断

### 第一步：查看Jupyter日志

Kernel崩溃时，首先查看日志：

**VSCode方法：**
1. 打开命令面板 (Cmd+Shift+P)
2. 输入 "Jupyter: Show Output"
3. 查看错误信息

**常见错误类型：**
```
MemoryError               → 内存不足
Killed                    → 系统强制终止（内存耗尽）
Bus error                 → GPU/库兼容问题
```

### 第二步：添加监控代码

在每个关键操作前后添加监控：

```python
# 在notebook开头导入
from memory_monitor import (
    MemoryMonitor, print_memory_usage,
    print_dataframe_info, MemoryProfiler
)

# 创建监控器
monitor = MemoryMonitor()

# 在关键操作前后
monitor.snapshot("数据加载前")
train_df = load_data()
monitor.snapshot("数据加载后")
monitor.report()  # 查看报告
```

---

## 内存监控工具使用

### 1. 基础监控

```python
# 快速查看当前内存
print_memory_usage("当前状态: ")

# 查看DataFrame详细信息
print_dataframe_info(train_df, "训练数据")

# 检查所有变量大小
check_variable_sizes(limit=10)
```

### 2. 操作追踪

```python
monitor = MemoryMonitor()

# 操作1
monitor.snapshot("加载数据前")
train_df = load_data()
monitor.snapshot("加载数据后")

# 操作2
monitor.snapshot("特征工程前")
train_df = add_features(train_df)
monitor.snapshot("特征工程后")

# 查看完整报告
monitor.report()
```

### 3. 代码块分析

```python
# 使用context manager自动分析
with MemoryProfiler():
    # 这里的代码会被自动分析
    train_df = load_data()
    model = train_model(train_df)
    # 自动显示内存使用报告
```

### 4. 装饰器监控

```python
from memory_monitor import monitor_memory

@monitor_memory("加载训练数据")
def load_data():
    # 函数代码
    return df

# 调用时自动显示内存使用
train_df = load_data()
```

---

## 常见问题诊断

### 问题1: 数据加载时崩溃

**症状：**
- `df = pd.read_parquet(...)` 后kernel死掉
- 数据文件很大（>1GB）

**诊断步骤：**
```python
# 1. 先检查文件大小
import os
file_size_gb = os.path.getsize('data.parquet') / 1024**3
print(f"文件大小: {file_size_gb:.2f} GB")

# 2. 估算加载后内存占用
# 通常parquet加载后是文件大小的2-5倍
estimated_mem = file_size_gb * 3
print(f"预计内存占用: {estimated_mem:.2f} GB")

# 3. 检查可用内存
import psutil
available_gb = psutil.virtual_memory().available / 1024**3
print(f"可用内存: {available_gb:.2f} GB")

if estimated_mem > available_gb * 0.8:
    print("⚠️  内存可能不足，需要采样或分批加载")
```

**解决方案：**
```python
# 方案1: 按日期/股票采样
def load_sampled_data(n_dates=10, n_symbols=10):
    # 只加载部分数据
    pass

# 方案2: 使用polars lazy loading
import polars as pl
df_lazy = pl.scan_parquet('data.parquet')
df = df_lazy.filter(...).collect()  # 只在filter后加载

# 方案3: 分块加载
chunk_size = 100000
chunks = []
for chunk in pd.read_parquet('data.parquet', chunksize=chunk_size):
    chunks.append(process_chunk(chunk))
    if len(chunks) >= max_chunks:  # 限制chunk数量
        break
df = pd.concat(chunks)
```

### 问题2: 模型训练时崩溃

**症状：**
- `model.fit()` 时kernel死掉
- 交叉验证时内存不断增长

**诊断步骤：**
```python
monitor = MemoryMonitor()

# 检查训练数据大小
monitor.snapshot("训练前")
print_dataframe_info(X_train, "X_train")
print_dataframe_info(y_train, "y_train")

# 监控训练过程
class TrainingCallback:
    def __init__(self, monitor):
        self.monitor = monitor
        self.iteration = 0

    def __call__(self, env):
        if self.iteration % 50 == 0:
            self.monitor.snapshot(f"迭代 {self.iteration}")
        self.iteration += 1

model.fit(
    X_train, y_train,
    callbacks=[TrainingCallback(monitor)]
)

monitor.report()
```

**解决方案：**
```python
# 1. 减少模型复杂度
params = {
    'n_estimators': 500,      # 减少树的数量
    'max_depth': 6,           # 减少深度
    'num_leaves': 64,         # 减少叶子数
}

# 2. 单折训练代替交叉验证
model.fit(X_train, y_train)  # 不使用CV

# 3. 使用early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(50)]
)

# 4. 训练后立即保存并清理
joblib.dump(model, 'model.pkl')
del model, X_train, y_train
gc.collect()
```

### 问题3: GPU导致的崩溃

**症状：**
- Mac上使用GPU时kernel崩溃
- 错误信息包含 "CUDA", "GPU", "metal"

**解决方案：**
```python
# 强制使用CPU
params = {
    'device': 'cpu',  # 明确设置
    'verbose': -1,
}
```

### 问题4: 内存泄漏

**症状：**
- 代码运行越来越慢
- 内存持续增长

**诊断：**
```python
# 追踪内存变化
monitor = MemoryMonitor()

for i in range(10):
    monitor.snapshot(f"循环 {i}")

    # 你的代码
    result = process_data(i)

    # 检查是否有内存泄漏
    monitor.alert(threshold_mb=500)  # 超过500MB就警告

monitor.report()
```

**解决方案：**
```python
# 1. 显式删除大对象
del large_df, temp_list
gc.collect()

# 2. 使用生成器代替列表
def process_items():
    for item in large_list:  # 不要: [process(x) for x in large_list]
        yield process(item)

# 3. 避免链式操作
# 不好: df = df.pipe(f1).pipe(f2).pipe(f3)
# 好:
df = f1(df)
del df
df = f2(df)
```

---

## 数据加载优化

### 推荐的内存安全加载模式

```python
def load_data_safe(file_path, max_memory_gb=4):
    """
    内存安全的数据加载
    """
    import polars as pl
    import psutil

    monitor = MemoryMonitor()
    monitor.snapshot("开始")

    # 1. 检查可用内存
    available = psutil.virtual_memory().available / 1024**3
    print(f"可用内存: {available:.2f} GB")

    # 2. 根据内存决定采样比例
    if available < 4:
        sample_ratio = 0.1
        n_dates = 5
    elif available < 8:
        sample_ratio = 0.3
        n_dates = 10
    else:
        sample_ratio = 1.0
        n_dates = None

    # 3. 使用polars lazy loading
    print(f"采样比例: {sample_ratio:.1%}")
    lazy_df = pl.scan_parquet(file_path)

    # 4. 只选择需要的列
    required_cols = ['feature_01', 'feature_02', ...]
    lazy_df = lazy_df.select(required_cols)

    # 5. 按日期过滤
    if n_dates:
        all_dates = lazy_df.select('date_id').unique().collect()
        selected_dates = all_dates[-n_dates:]
        lazy_df = lazy_df.filter(pl.col('date_id').is_in(selected_dates))

    # 6. 转换为pandas
    df = lazy_df.collect().to_pandas()

    monitor.snapshot("加载完成")
    monitor.report()

    return df
```

---

## 模型训练优化

### 内存高效的训练模式

```python
def train_safe(train_df, valid_df):
    """内存安全的训练"""
    monitor = MemoryMonitor()
    monitor.snapshot("训练前")

    # 1. 准备数据
    feature_cols = [f'feature_{i:02d}' for i in range(79)]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    w_train = train_df['weight']

    X_valid = valid_df[feature_cols]
    y_valid = valid_df['target']
    w_valid = valid_df['weight']

    print_dataframe_info(X_train, "训练特征")

    # 2. 删除原始数据（已提取特征）
    del train_df, valid_df
    gc.collect()
    monitor.snapshot("删除原始数据后")

    # 3. 训练模型（使用早停）
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        num_leaves=64,
        device='cpu',
        verbose=-1,
    )

    model.fit(
        X_train, y_train, w_train,
        eval_set=[(X_valid, y_valid, w_valid)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )

    monitor.snapshot("训练完成")
    monitor.report()

    # 4. 立即保存模型
    joblib.dump(model, 'model.pkl')
    print("模型已保存")

    # 5. 清理
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()

    return model
```

---

## 调试清单

### 开始训练前

- [ ] 检查数据文件大小
- [ ] 检查系统可用内存
- [ ] 估算加载后内存占用
- [ ] 决定是否需要采样

### 数据加载时

- [ ] 使用 `MemoryMonitor` 追踪
- [ ] 只选择需要的列
- [ ] 考虑使用polars lazy loading
- [ ] 加载后检查DataFrame内存占用
- [ ] 及时删除不需要的变量

### 训练前

- [ ] 检查训练数据大小
- [ ] 考虑减少模型复杂度
- [ ] 启用early stopping
- [ ] 使用CPU代替GPU（Mac）
- [ ] 准备监控回调

### 训练后

- [ ] 立即保存模型
- [ ] 删除训练数据
- [ ] 强制垃圾回收
- [ ] 检查最终内存使用

---

## 快速参考：内存估算公式

```
# DataFrame内存估算
内存(GB) ≈ 行数 × 列数 × 每个元素大小 × 1.5

# float64: 8字节
# float32: 4字节
# int64: 8字节
# int32: 4字节
# object/str: 变长（通常更大）

示例:
100万行 × 80列 × 8字节 × 1.5 / 1GB ≈ 0.96 GB
```

## 推荐的Notebook结构

```python
# Cell 1: 导入和设置
from memory_monitor import MemoryMonitor, print_memory_usage
monitor = MemoryMonitor()

# Cell 2: 加载数据
monitor.snapshot("加载前")
train_df = load_data()
monitor.snapshot("加载后")
monitor.report()
print_dataframe_info(train_df, "训练数据")

# Cell 3: 特征工程
monitor.snapshot("特征工程前")
train_df = add_features(train_df)
del monitor.snapshots[:]  # 清空旧快照
monitor.snapshot("特征工程后")

# Cell 4: 训练
with MemoryProfiler():
    model = train_model(train_df)

# Cell 5: 清理
del train_df
gc.collect()
print_memory_usage("清理后: ")
```

---

## 总结

**最重要的三个原则：**

1. **监控一切** - 在关键操作前后都检查内存
2. **尽早释放** - 用完就删除，不要等到最后
3. **保守估计** - 如果内存紧张，假设需要3倍的预期内存
