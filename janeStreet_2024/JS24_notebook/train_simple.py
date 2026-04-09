#!/usr/bin/env python3
"""
LightGBM训练脚本 - 最简化版本

使用方法:
    python train_simple.py
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import gc
import psutil
import os

import lightgbm as lgb
import joblib


# ============== 配置参数 ==============

# 数据路径
DATA_DIR = Path('./processed_data')
TRAIN_FILE = DATA_DIR / 'training.parquet'
VALID_FILE = DATA_DIR / 'validation.parquet'
MODEL_DIR = Path('./models')

# 特征列名（79个feature + 9个lag）
FEATURES = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{i}_lag_1" for i in range(9)]

# 数据采样参数
N_TRAIN_DAYS = 600   # 训练集加载多少天
N_VALID_DAYS = 100   # 验证集加载多少天
N_SYMBOLS = 39       # 加载多少个股票

# 模型参数
N_TREES = 500        # 树的数量
MAX_DEPTH = 6        # 树的最大深度
NUM_LEAVES = 64      # 叶子节点数
LEARNING_RATE = 0.05 # 学习率
EARLY_STOPPING = 50  # 早停轮数


# ============== 工具函数 ==============

def print_memory(tag=""):
    """打印当前内存使用"""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"{tag}内存: {mem_gb:.2f} GB")


def weighted_r2(y_true, y_pred, weights):
    """
    加权R²得分

    公式: R² = 1 - Σ(w*(y-ŷ)²) / Σ(w*y²)
    """
    numerator = np.average((y_true - y_pred)**2, weights=weights)
    denominator = np.average(y_true**2, weights=weights) + 1e-38
    return 1 - numerator / denominator


def lgb_metric(y_true, y_pred, weights):
    """LightGBM格式的加权R²评估函数"""
    r2 = weighted_r2(y_true, y_pred, weights)
    return 'r2', r2, True  # (名称, 值, 越大越好)


# ============== 主训练函数 ==============

def main():
    """主训练函数"""

    print("="*60)
    print("LightGBM训练 - 最简化版本")
    print("="*60)

    print(f"\n配置参数:")
    print(f"  特征数: {len(FEATURES)}")
    print(f"  训练集: {N_TRAIN_DAYS}天 x {N_SYMBOLS}股票")
    print(f"  验证集: {N_VALID_DAYS}天 x {N_SYMBOLS}股票")
    print(f"  模型: {N_TREES}棵树, 深度{MAX_DEPTH}, 学习率{LEARNING_RATE}")

    # 创建模型输出目录
    MODEL_DIR.mkdir(exist_ok=True)

    # 记录初始内存
    print_memory("开始: ")

    # ===== 加载训练数据 =====
    print("\n[1/2] 加载训练数据...")
    train_lazy = pl.scan_parquet(TRAIN_FILE)

    # 采样：只取最后N_TRAIN_DAYS天、前N_SYMBOLS个股票
    all_dates = train_lazy.select('date_id').unique().collect().to_series().to_list()
    all_dates.sort()
    selected_dates = all_dates[-N_TRAIN_DAYS:]

    all_symbols = train_lazy.select('symbol_id').unique().collect().to_series().to_list()
    selected_symbols = sorted(all_symbols)[:N_SYMBOLS]

    print(f"  日期范围: {selected_dates[0]}-{selected_dates[-1]} ({len(selected_dates)}天)")
    print(f"  股票数: {len(selected_symbols)}个")

    # 过滤数据
    train_lazy = train_lazy.filter(
        pl.col('date_id').is_in(selected_dates) &
        pl.col('symbol_id').is_in(selected_symbols)
    )

    # 填充缺失值并转换为pandas
    needed_cols = FEATURES + ['responder_6', 'weight', 'date_id', 'symbol_id']
    train_lazy = train_lazy.select(needed_cols)
    train_lazy = train_lazy.with_columns([pl.col(f).fill_null(0) for f in FEATURES])

    train_df = train_lazy.collect().to_pandas()
    print(f"  ✓ 训练数据: {train_df.shape}")
    print_memory("训练数据加载后: ")

    # ===== 加载验证数据 =====
    print("\n[2/2] 加载验证数据...")
    valid_lazy = pl.scan_parquet(VALID_FILE)

    # 采样：只取最后N_VALID_DAYS天
    all_dates = valid_lazy.select('date_id').unique().collect().to_series().to_list()
    all_dates.sort()
    selected_dates = all_dates[-N_VALID_DAYS:]

    valid_lazy = valid_lazy.filter(
        pl.col('date_id').is_in(selected_dates) &
        pl.col('symbol_id').is_in(selected_symbols)
    )

    valid_lazy = valid_lazy.select(needed_cols)
    valid_lazy = valid_lazy.with_columns([pl.col(f).fill_null(0) for f in FEATURES])

    valid_df = valid_lazy.collect().to_pandas()
    print(f"  ✓ 验证数据: {valid_df.shape}")
    print_memory("验证数据加载后: ")
    # pd.set_option('display.max_columns', None)
    # print(train_df.dtypes)
    # print(valid_df.dtypes)
    train_df.info(memory_usage='deep')
    valid_df.info(memory_usage='deep')

    # ===== 准备训练数据 =====
    print("\n准备训练数据...")

    X_train = train_df[FEATURES]
    y_train = train_df['responder_6']
    w_train = train_df['weight']

    X_valid = valid_df[FEATURES]
    y_valid = valid_df['responder_6']
    w_valid = valid_df['weight']

    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_valid.shape}")

    # 删除原始DataFrame，释放内存
    del train_df, valid_df
    gc.collect()

    print_memory("准备完成后: ")

    # ===== 训练模型 =====
    print("\n开始训练...")
    print("="*60)

    # 创建模型
    model = lgb.LGBMRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        objective='regression',
        metric='None',
        device='cpu',
        verbose=-1,
    )

    # 训练
    model.fit(
        X_train, y_train, w_train,
        eval_set=[(X_valid, y_valid, w_valid)],
        eval_metric=lgb_metric,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING),
            lgb.log_evaluation(50)
        ]
    )

    print_memory("训练后: ")

    # ===== 评估结果 =====
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_r2 = weighted_r2(y_train, train_pred, w_train)
    valid_r2 = weighted_r2(y_valid, valid_pred, w_valid)

    print("\n" + "="*60)
    print("训练结果:")
    print("="*60)
    print(f"训练集 R²: {train_r2:.6f}")
    print(f"验证集 R²: {valid_r2:.6f}")
    print(f"实际训练树数: {model.booster_.num_trees()}")
    print("="*60)

    # ===== 保存模型 =====
    model_path = MODEL_DIR / 'simple_lgb_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n模型已保存: {model_path}")

    # 清理内存
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()

    print_memory("清理后: ")

    # ===== 总结 =====
    print(f"""
========================================
训练流程总结
========================================

1. 加载数据
   - 训练集: {N_TRAIN_DAYS}天 x {N_SYMBOLS}股票
   - 验证集: {N_VALID_DAYS}天 x {N_SYMBOLS}股票

2. 训练模型
   - 最多{N_TREES}棵树
   - 最大深度{MAX_DEPTH}
   - 学习率{LEARNING_RATE}

3. 结果
   - 训练集R²: {train_r2:.4f}
   - 验证集R²: {valid_r2:.4f}

4. 模型已保存
   - 文件: simple_lgb_model.pkl
========================================
""")


if __name__ == "__main__":
    main()
