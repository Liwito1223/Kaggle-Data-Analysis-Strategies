# Jane Street 实时市场数据预测 - 学习项目

## 项目概述

这是一个基于Kaggle竞赛 [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) 的学习项目。

### 竞赛目标

预测金融市场中`responder_6`的值，这是一个回归任务。

### 数据说明

- **特征**: 79个市场特征 (feature_00 到 feature_78)
- **目标**: responder_6
- **结构**: 按日期(date_id)、时间(time_id)、股票(symbol_id)分组
- **权重**: 每个样本有权重，用于加权评估

## 项目结构

```
JS24/
├── README.md                                    # 本文件
├── 01_EDA_exploratory_data_analysis.ipynb      # 探索性数据分析
├── 02_Lag_Engineering.ipynb                    # 滞后特征工程
├── 03_Training_LightGBM.ipynb                  # LightGBM模型训练
├── 04_Training_XGBoost.ipynb                   # XGBoost模型训练
├── 05_Training_NeuralNetwork.ipynb             # 神经网络模型训练
├── processed_data/                             # 处理后的数据（运行滞后特征工程后生成）
│   ├── training.parquet
│   └── validation.parquet
└── models/                                     # 训练好的模型（运行训练笔记本后生成）
    ├── lgb_model_*.pkl
    ├── xgb_model_*.pkl
    └── nn_model_*.pt
```

## 笔记本说明

### 1. EDA (探索性数据分析)

**文件**: `01_EDA_exploratory_data_analysis.ipynb`

**内容**:
- 数据结构理解
- 目标变量分析
- 特征分布和相关性分析
- 权重分析
- 数据质量检查

**学习要点**:
- 了解金融时间序列数据的特点
- 掌握EDA的基本方法和技巧
- 学习内存优化策略

### 2. 滞后特征工程

**文件**: `02_Lag_Engineering.ipynb`

**内容**:
- 什么是滞后特征
- 如何创建滞后特征
- 模拟推理时的数据处理

**学习要点**:
- 理解滞后特征的重要性
- 掌握时间序列特征工程方法
- 确保训练和推理的数据格式一致

### 3. LightGBM训练

**文件**: `03_Training_LightGBM.ipynb`

**为什么选择LightGBM**:
- 训练速度快
- 内存效率高
- 在表格数据上表现优秀

**学习要点**:
- LightGBM的基本使用
- 自定义评估指标（加权R²）
- 交叉验证训练
- 特征重要性分析

### 4. XGBoost训练

**文件**: `04_Training_XGBoost.ipynb`

**为什么选择XGBoost**:
- 稳定可靠
- 生产环境广泛使用
- 功能丰富

**学习要点**:
- XGBoost的基本使用
- 超参数调优
- 学习曲线分析
- 与LightGBM的对比

### 5. 神经网络训练

**文件**: `05_Training_NeuralNetwork.ipynb`

**为什么选择神经网络**:
- 能够学习复杂的非线性关系
- 自动特征学习
- 端到端训练

**学习要点**:
- PyTorch基础
- MLP网络设计
- 自定义数据集和数据加载器
- 训练循环实现
- 学习率调度和早停

## 核心概念

### 评估指标：加权R²

```
R² = 1 - Σ(wi × (yi - ŷi)²) / Σ(wi × yi²)
```

其中:
- yi: 真实值
- ŷi: 预测值
- wi: 样本权重

**解释**:
- R² = 1: 完美预测
- R² = 0: 相当于预测均值
- R² < 0: 比预测均值还差

### 滞后特征 (Lag Features)

**概念**: 将过去时间点的变量值作为当前时间点的特征

**在这个项目中**:
- `responder_X_lag_1`: 前一交易日最后时间点的responder_X值
- API会在推理时提供这些滞后值
- 训练时需要自己创建

### 时间序列交叉验证

**为什么需要**:
- 避免使用未来数据（look-ahead bias）
- 模拟真实的推理场景
- 更准确的模型评估

**方法**:
- 按时间划分验证集
- 确保训练集的时间早于验证集

## 模型对比

| 模型 | 训练速度 | 预测速度 | 内存占用 | 调参难度 | 适用场景 |
|------|---------|---------|---------|---------|----------|
| LightGBM | 很快 | 很快 | 低 | 中 | 表格数据，快速迭代 |
| XGBoost | 快 | 快 | 中 | 中 | 表格数据，稳定性优先 |
| 神经网络 | 慢 | 快 | 高 | 高 | 复杂非线性关系 |

## 运行指南

### 环境要求

```bash
# 核心依赖
numpy
pandas
polars
lightgbm
xgboost
torch
scikit-learn
matplotlib
```

### 运行顺序

1. **EDA笔记本** - 了解数据
2. **滞后特征工程** - 准备训练数据
3. **选择模型训练** - 选择一个或多个模型进行训练

### 内存管理

所有笔记本都经过优化，确保在10GB内存内运行：

- 使用polars进行高效数据加载
- 分区处理数据
- 及时释放内存
- float32代替float64

## 学习路径建议

### 初学者

1. 先运行EDA，理解数据
2. 运行LightGBM训练（最快）
3. 理解评估指标和特征重要性

### 进阶

1. 尝试XGBoost，对比性能
2. 学习滞后特征工程
3. 尝试神经网络

### 高级

1. 超参数调优
2. 模型集成
3. 特征工程优化

## 常见问题

### Q1: 为什么滞后特征有缺失值？

A: 第一天的数据没有前一天的数据，用0填充。

### Q2: 训练和推理的数据格式一致吗？

A: 应该一致！滞后特征工程确保了这一点。

### Q3: 如何选择模型？

A: 建议都尝试，然后选择表现最好的或进行集成。

### Q4: 内存不足怎么办？

A: 尝试：
- 减小batch_size
- 处理更少的数据分区
- 使用更小的模型

## 下一步

1. **模型优化**:
   - 超参数搜索
   - 特征选择
   - 模型集成

2. **特征工程**:
   - 创建更多滞后特征
   - 特征交叉
   - 技术指标

3. **高级架构**:
   - Transformer
   - TabNet
   - 深度神经网络

## 资源

- [Kaggle竞赛页面](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- [讨论区](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion)
- [参考解决方案](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/code)

## 作者

本项目由AI助手创建，用于学习量化交易和机器学习。

## 许可

本项目仅用于学习目的。
