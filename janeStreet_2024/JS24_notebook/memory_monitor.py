"""
Jupyter Notebook 内存和资源监控工具

用于调试kernel崩溃问题，追踪内存使用情况
"""

import gc
import sys
import time
import psutil
import os
from functools import wraps
from IPython.display import display, HTML
import pandas as pd


class MemoryMonitor:
    """
    内存监控类 - 追踪内存使用情况
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        self.baseline = self._get_memory_info()

    def _get_memory_info(self):
        """获取当前内存信息"""
        mem = self.process.memory_info()
        return {
            'rss_mb': mem.rss / 1024**2,  # 实际物理内存
            'vms_mb': mem.vms / 1024**2,  # 虚拟内存
        }

    def snapshot(self, label=""):
        """
        创建一个内存快照

        参数:
            label: 快照标签（描述当前操作）
        """
        current = self._get_memory_info()
        snapshot = {
            'label': label,
            'rss_mb': current['rss_mb'],
            'vms_mb': current['vms_mb'],
            'rss_delta_mb': current['rss_mb'] - self.baseline['rss_mb'],
            'timestamp': time.time()
        }
        self.snapshots.append(snapshot)
        return snapshot

    def report(self, last_n=None):
        """
        显示内存使用报告

        参数:
            last_n: 只显示最后n个快照，None表示显示全部
        """
        if not self.snapshots:
            print("没有快照数据")
            return

        snapshots = self.snapshots[-last_n:] if last_n else self.snapshots

        df = pd.DataFrame(snapshots)
        df['rss_gb'] = df['rss_mb'] / 1024
        df['rss_delta_gb'] = df['rss_delta_mb'] / 1024

        display_df = df[['label', 'rss_gb', 'rss_delta_gb']].copy()
        display_df.columns = ['操作', 'RSS内存(GB)', '增量(GB)']
        display_df = display_df.round(3)

        print("\n" + "="*60)
        print("内存使用报告")
        print("="*60)
        display(display_df)

        # 找出内存增长最多的操作
        df_sorted = df.sort_values('rss_delta_mb', ascending=False)
        print("\n内存增长最多的操作:")
        for _, row in df_sorted.head(5).iterrows():
            print(f"  {row['label']}: +{row['rss_delta_mb']:.1f} MB")

    def alert(self, threshold_mb=1000):
        """
        如果内存增长超过阈值，显示警告

        参数:
            threshold_mb: 内存增长阈值(MB)
        """
        if not self.snapshots:
            return

        current = self.snapshots[-1]
        if current['rss_delta_mb'] > threshold_mb:
            print(f"\n⚠️  警告: 内存增长了 {current['rss_delta_mb']:.1f} MB")
            print(f"   当前总内存: {current['rss_mb']:.1f} MB ({current['rss_mb']/1024:.2f} GB)")


def monitor_memory(label=""):
    """
    装饰器：监控函数的内存使用

    使用方法:
    @monitor_memory("加载数据")
    def load_data():
        ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            monitor.snapshot(f"开始: {label}")

            try:
                result = func(*args, **kwargs)
                monitor.snapshot(f"完成: {label}")
                monitor.report(last_n=2)
                return result
            except Exception as e:
                monitor.snapshot(f"失败: {label}")
                monitor.report(last_n=2)
                raise e

        return wrapper
    return decorator


def print_memory_usage(prefix=""):
    """
    打印当前内存使用情况

    参数:
        prefix: 前缀文本
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info()

    rss_gb = mem.rss / 1024**3
    vms_gb = mem.vms / 1024**3

    print(f"{prefix}内存使用: RSS={rss_gb:.2f} GB, VMS={vms_gb:.2f} GB")

    # 获取系统总内存信息
    sys_mem = psutil.virtual_memory()
    print(f"{prefix}系统内存: {sys_mem.percent}% 已使用 "
          f"({sys_mem.used/1024**3:.1f}GB / {sys_mem.total/1024**3:.1f}GB)")


def get_dataframe_memory(df):
    """
    获取DataFrame的内存使用情况

    参数:
        df: pandas DataFrame

    返回:
        内存使用信息字典
    """
    mem = df.memory_usage(deep=True)
    total_mb = mem.sum() / 1024**2
    total_gb = total_mb / 1024

    # 按列统计
    col_mem = pd.DataFrame({
        '列名': mem.index,
        '内存_MB': mem.values / 1024**2
    }).sort_values('内存_MB', ascending=False)

    return {
        'total_mb': total_mb,
        'total_gb': total_gb,
        'by_column': col_mem,
        'shape': df.shape,
        'n_rows': df.shape[0],
        'n_cols': df.shape[1]
    }


def print_dataframe_info(df, name="DataFrame"):
    """
    打印DataFrame的详细信息

    参数:
        df: pandas DataFrame
        name: DataFrame名称
    """
    info = get_dataframe_memory(df)

    print(f"\n{'='*60}")
    print(f"{name} 信息")
    print(f"{'='*60}")
    print(f"形状: {info['n_rows']:,} 行 × {info['n_cols']} 列")
    print(f"总内存: {info['total_mb']:.2f} MB ({info['total_gb']:.3f} GB)")
    print(f"每行平均: {info['total_mb']/info['n_rows']*1000:.2f} KB")

    print(f"\n内存占用最高的列 (前10):")
    display(info['by_column'].head(10))


def force_gc():
    """
    强制垃圾回收并打印内存变化
    """
    print("\n强制垃圾回收...")
    before = print_memory_usage("回收前: ")

    gc.collect()

    after = print_memory_usage("回收后: ")


def check_variable_sizes(limit=10):
    """
    检查当前环境中所有变量的内存占用

    参数:
        limit: 显示前n个最大的变量
    """
    import sys

    # 获取全局变量
    global_vars = globals()

    var_sizes = []
    for name, obj in global_vars.items():
        if not name.startswith('_'):
            try:
                size = sys.getsizeof(obj)
                # 对于DataFrame/Array，获取更准确的大小
                if hasattr(obj, 'memory_usage'):
                    size = obj.memory_usage(deep=True).sum()
                elif hasattr(obj, 'nbytes'):
                    size = obj.nbytes

                var_sizes.append({
                    'name': name,
                    'type': type(obj).__name__,
                    'size_mb': size / 1024**2
                })
            except:
                pass

    df = pd.DataFrame(var_sizes).sort_values('size_mb', ascending=False)

    print(f"\n内存占用最大的变量 (前{limit}个):")
    display(df.head(limit))

    return df


class MemoryProfiler:
    """
    内存分析器 - 自动追踪代码块的内存使用
    """

    def __init__(self):
        self.monitor = MemoryMonitor()

    def __enter__(self):
        self.monitor.snapshot("代码块开始")
        print_memory_usage("开始: ")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.snapshot("代码块结束")
        print_memory_usage("结束: ")
        self.monitor.report()

        # 自动清理
        gc.collect()

        if exc_type is MemoryError:
            print("\n⚠️  检测到 MemoryError!")
            print("建议:")
            print("  1. 减少数据量（采样）")
            print("  2. 使用更节省内存的数据类型")
            print("  3. 分批处理数据")
            print("  4. 删除不需要的变量")


# 便捷函数
def quick_profile(func, *args, **kwargs):
    """
    快速分析函数的内存使用

    使用:
    result = quick_profile(load_data)
    """
    print(f"\n分析函数: {func.__name__}")
    print("-" * 60)

    monitor = MemoryMonitor()
    monitor.snapshot("开始")

    try:
        result = func(*args, **kwargs)
        monitor.snapshot("完成")
        monitor.report()
        return result
    except Exception as e:
        monitor.snapshot(f"错误: {type(e).__name__}")
        monitor.report()
        raise


# ========== 使用示例 ==========

if __name__ == "__main__":
    print("内存监控工具已加载")
    print("\n使用方法:")
    print("1. monitor = MemoryMonitor()  # 创建监控器")
    print("2. monitor.snapshot('操作名称')  # 创建快照")
    print("3. monitor.report()  # 查看报告")
    print("\n或者使用:")
    print("- print_memory_usage()  # 快速查看当前内存")
    print("- print_dataframe_info(df)  # 查看DataFrame信息")
    print("- with MemoryProfiler():  # 自动分析代码块")
