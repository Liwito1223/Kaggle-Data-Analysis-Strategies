"""
内存优化的数据加载工具

这个模块提供了内存安全的数据加载函数，确保在有限内存（<10GB）的情况下处理大数据集。
"""

import polars as pl
import pandas as pd
from pathlib import Path
import gc


def check_memory_usage():
    """检查当前内存使用情况"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print(f"当前进程内存使用:")
    print(f"  RSS: {mem_info.rss / 1024**3:.2f} GB")
    print(f"  VMS: {mem_info.vms / 1024**3:.2f} GB")

    return mem_info.rss / 1024**3


def load_data_memory_safe(
    train_file,
    valid_file,
    feature_cols,
    target_col='responder_6',
    weight_col='weight',
    max_memory_gb=8,
    sample_ratio=None,
    use_lazy=True
):
    """
    内存安全的数据加载函数

    参数:
        train_file: 训练数据文件路径
        valid_file: 验证数据文件路径
        feature_cols: 特征列名列表
        target_col: 目标列名
        weight_col: 权重列名
        max_memory_gb: 最大允许内存使用（GB）
        sample_ratio: 采样比例（0-1），如果为None则自动计算
        use_lazy: 是否使用lazy loading（推荐）

    返回:
        train_df, valid_df: pandas DataFrame
    """

    print("=" * 50)
    print("内存安全数据加载")
    print("=" * 50)

    # 检查初始内存
    initial_mem = check_memory_usage()

    # 需要的列
    required_cols = feature_cols + [target_col, weight_col, 'date_id', 'symbol_id']

    # ========== 加载训练数据 ==========
    print("\n[1/4] 扫描训练数据...")

    if use_lazy:
        # 使用lazy loading，不立即加载到内存
        train_lazy = pl.scan_parquet(train_file)

        # 只选择需要的列（使用高效方法）
        schema = train_lazy.collect_schema()
        available_cols = [col for col in required_cols if col in schema.names()]
        train_lazy = train_lazy.select(available_cols)

        # 获取数据信息
        n_rows_train = train_lazy.select(pl.len()).collect().item()
        print(f"  训练数据总行数: {n_rows_train:,}")

        # 计算采样比例
        if sample_ratio is None:
            # 根据内存限制自动计算采样比例
            available_memory = max_memory_gb - initial_mem
            if available_memory < 2:
                sample_ratio = 0.1  # 只使用10%的数据
                print(f"  ⚠️  可用内存较少，使用10%采样")
            elif available_memory < 4:
                sample_ratio = 0.3  # 使用30%的数据
                print(f"  ⚠️  可用内存中等，使用30%采样")
            else:
                sample_ratio = 1.0  # 使用全部数据
                print(f"  ✓ 可用内存充足，使用全部数据")
        else:
            print(f"  使用指定的采样比例: {sample_ratio:.1%}")

        # 采样策略：按日期采样（保持时间序列的连续性）
        if sample_ratio < 1.0:
            print(f"\n[2/4] 按日期采样训练数据...")

            # 获取所有日期
            all_dates = train_lazy.select('date_id').unique().collect().to_series().to_list()
            all_dates.sort()

            # 选择连续的日期段
            n_dates_to_keep = int(len(all_dates) * sample_ratio)
            selected_dates = all_dates[-n_dates_to_keep:]  # 保留最新的日期

            print(f"  原始日期数: {len(all_dates)}")
            print(f"  采样后日期数: {n_dates_to_keep}")
            print(f"  日期范围: {selected_dates[0]} 到 {selected_dates[-1]}")

            # 过滤数据
            train_lazy = train_lazy.filter(pl.col('date_id').is_in(selected_dates))

        # 填充缺失值
        print(f"\n[3/4] 处理缺失值...")
        train_lazy = train_lazy.with_columns([
            pl.col(col).fill_null(0) for col in feature_cols if col in available_cols
        ])

        # 转换为pandas
        print(f"\n[4/4] 转换为pandas...")
        train_pl = train_lazy.collect()
        train_df = train_pl.to_pandas()

        print(f"  ✓ 训练数据加载完成: {train_df.shape}")
        print(f"  内存使用: {check_memory_usage():.2f} GB")

        # 清理
        del train_lazy, train_pl
        gc.collect()
    else:
        # 直接加载（不推荐）
        print("  ⚠️  使用直接加载模式（可能消耗大量内存）")
        train_pl = pl.read_parquet(train_file)
        available_cols = [col for col in required_cols if col in train_pl.columns]
        train_pl = train_pl.select(available_cols)

        for col in feature_cols:
            if col in train_pl.columns:
                train_pl = train_pl.with_columns([
                    pl.col(col).fill_null(0)
                ])

        train_df = train_pl.to_pandas()
        print(f"  ✓ 训练数据加载完成: {train_df.shape}")
        del train_pl
        gc.collect()

    # ========== 加载验证数据 ==========
    print(f"\n[1/3] 扫描验证数据...")

    if valid_file and Path(valid_file).exists():
        if use_lazy:
            valid_lazy = pl.scan_parquet(valid_file)

            # 使用高效方法检查列
            schema = valid_lazy.collect_schema()
            available_cols = [col for col in required_cols if col in schema.names()]
            valid_lazy = valid_lazy.select(available_cols)

            n_rows_valid = valid_lazy.select(pl.len()).collect().item()
            print(f"  验证数据总行数: {n_rows_valid:,}")

            # 验证集也采样（与训练集相同的比例）
            if sample_ratio < 1.0:
                print(f"\n[2/3] 采样验证数据...")

                all_dates = valid_lazy.select('date_id').unique().collect().to_series().to_list()
                all_dates.sort()

                n_dates_to_keep = int(len(all_dates) * sample_ratio)
                selected_dates = all_dates[-n_dates_to_keep:]

                valid_lazy = valid_lazy.filter(pl.col('date_id').is_in(selected_dates))

                print(f"  采样后日期数: {n_dates_to_keep}")

            # 填充缺失值
            print(f"\n[3/3] 处理缺失值并转换...")
            valid_lazy = valid_lazy.with_columns([
                pl.col(col).fill_null(0) for col in feature_cols if col in available_cols
            ])

            valid_pl = valid_lazy.collect()
            valid_df = valid_pl.to_pandas()

            print(f"  ✓ 验证数据加载完成: {valid_df.shape}")
            print(f"  内存使用: {check_memory_usage():.2f} GB")

            del valid_lazy, valid_pl
            gc.collect()
        else:
            valid_pl = pl.read_parquet(valid_file)
            available_cols = [col for col in required_cols if col in valid_pl.columns]
            valid_pl = valid_pl.select(available_cols)

            for col in feature_cols:
                if col in valid_pl.columns:
                    valid_pl = valid_pl.with_columns([
                        pl.col(col).fill_null(0)
                    ])

            valid_df = valid_pl.to_pandas()
            print(f"  ✓ 验证数据加载完成: {valid_df.shape}")
            del valid_pl
            gc.collect()
    else:
        print("  验证数据文件不存在")
        valid_df = None

    # ========== 最终检查 ==========
    print("\n" + "=" * 50)
    print("数据加载完成！")
    print("=" * 50)
    print(f"训练数据: {train_df.shape}")
    print(f"验证数据: {valid_df.shape if valid_df is not None else 'N/A'}")
    print(f"最终内存使用: {check_memory_usage():.2f} GB")
    print("=" * 50)

    return train_df, valid_df


def load_data_minimal(
    train_file,
    valid_file,
    feature_cols,
    target_col='responder_6',
    weight_col='weight',
    n_samples=1000000
):
    """
    最小内存加载函数 - 只加载样本数据用于快速测试

    参数:
        n_samples: 要加载的样本数量

    返回:
        train_df, valid_df: pandas DataFrame
    """

    print("=" * 50)
    print(f"快速测试模式 - 加载{n_samples:,}个样本")
    print("=" * 50)

    required_cols = feature_cols + [target_col, weight_col, 'date_id', 'symbol_id']

    # 加载训练数据
    print("\n加载训练数据...")
    train_lazy = pl.scan_parquet(train_file)
    available_cols = [col for col in required_cols if col in train_lazy.columns]
    train_lazy = train_lazy.select(available_cols)

    # 采样
    train_lazy = train_lazy.head(n_samples)
    train_lazy = train_lazy.with_columns([
        pl.col(col).fill_null(0) for col in feature_cols if col in available_cols
    ])

    train_pl = train_lazy.collect()
    train_df = train_pl.to_pandas()

    print(f"  ✓ 训练数据: {train_df.shape}")

    # 加载验证数据
    if valid_file and Path(valid_file).exists():
        print("\n加载验证数据...")
        valid_lazy = pl.scan_parquet(valid_file)
        available_cols = [col for col in required_cols if col in valid_lazy.columns]
        valid_lazy = valid_lazy.select(available_cols)

        # 采样（较少的样本）
        valid_lazy = valid_lazy.head(n_samples // 5)
        valid_lazy = valid_lazy.with_columns([
            pl.col(col).fill_null(0) for col in feature_cols if col in available_cols
        ])

        valid_pl = valid_lazy.collect()
        valid_df = valid_pl.to_pandas()

        print(f"  ✓ 验证数据: {valid_df.shape}")
    else:
        valid_df = None

    print(f"\n最终内存使用: {check_memory_usage():.2f} GB")
    print("=" * 50)

    return train_df, valid_df


# 使用示例
if __name__ == "__main__":
    # 示例配置
    FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{i}_lag_1" for i in range(9)]

    TRAIN_FILE = Path("./processed_data/training.parquet")
    VALID_FILE = Path("./processed_data/validation.parquet")

    # 方式1: 自动内存管理（推荐）
    train_df, valid_df = load_data_memory_safe(
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        feature_cols=FEATURE_COLS,
        max_memory_gb=8
    )

    # 方式2: 快速测试模式
    # train_df, valid_df = load_data_minimal(
    #     train_file=TRAIN_FILE,
    #     valid_file=VALID_FILE,
    #     feature_cols=FEATURE_COLS,
    #     n_samples=100000
    # )
