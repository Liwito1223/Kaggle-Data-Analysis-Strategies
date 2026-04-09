"""
实验跟踪器 - 系统化管理参数实验

功能:
1. 自动记录每次实验的参数和结果
2. 保存实验历史到JSON文件
3. 提供实验结果查询和对比功能
4. 生成实验报告和可视化
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ExperimentResult:
    """单次实验结果"""
    experiment_id: str
    timestamp: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    model_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class ExperimentTracker:
    """实验跟踪器"""

    def __init__(self, experiment_dir: Path = Path("./experiments")):
        """
        初始化实验跟踪器

        参数:
            experiment_dir: 实验数据保存目录
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

        # 实验历史文件
        self.history_file = self.experiment_dir / "experiment_history.json"
        self.history = self._load_history()

        # 当前实验ID
        self.current_experiment_id = None
        self.current_start_time = None

    def _load_history(self) -> List[Dict]:
        """加载实验历史"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_history(self):
        """保存实验历史"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _generate_experiment_id(self, params: Dict) -> str:
        """生成唯一的实验ID"""
        # 使用参数的哈希值作为ID
        params_str = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.md5(params_str.encode())
        return hash_obj.hexdigest()[:8]

    def start_experiment(self, params: Dict[str, Any], tags: List[str] = None) -> str:
        """
        开始一个新实验

        参数:
            params: 实验参数
            tags: 实验标签（可选）

        返回:
            实验ID
        """
        self.current_experiment_id = self._generate_experiment_id(params)
        self.current_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"开始实验: {self.current_experiment_id}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if tags:
            print(f"标签: {', '.join(tags)}")
        print(f"{'='*60}")
        print("\n参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        return self.current_experiment_id

    def log_metric(self, key: str, value: float):
        """
        记录一个指标

        参数:
            key: 指标名称
            value: 指标值
        """
        print(f"  {key}: {value:.6f}")

    def end_experiment(
        self,
        metrics: Dict[str, float],
        model_path: str = None,
        metadata: Dict[str, Any] = None
    ) -> ExperimentResult:
        """
        结束当前实验并记录结果

        参数:
            metrics: 评估指标字典
            model_path: 模型保存路径（可选）
            metadata: 额外的元数据（可选）

        返回:
            ExperimentResult对象
        """
        if self.current_experiment_id is None:
            raise ValueError("没有正在运行的实验。请先调用start_experiment()")

        # 计算运行时间
        elapsed_time = time.time() - self.current_start_time

        print(f"\n实验完成!")
        print(f"运行时间: {elapsed_time:.2f}秒")
        print("\n最终指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # 创建实验结果
        result = ExperimentResult(
            experiment_id=self.current_experiment_id,
            timestamp=datetime.now().isoformat(),
            params={},  # 需要从外部传入
            metrics=metrics,
            metadata=metadata or {},
            model_path=model_path
        )

        # 保存到历史
        self.history.append(result.to_dict())
        self._save_history()

        # 重置当前实验
        self.current_experiment_id = None
        self.current_start_time = None

        return result

    def get_history_df(self) -> pd.DataFrame:
        """获取实验历史DataFrame"""
        if not self.history:
            return pd.DataFrame()

        # 展开params和metrics
        rows = []
        for exp in self.history:
            row = {
                'experiment_id': exp['experiment_id'],
                'timestamp': exp['timestamp'],
                **exp.get('params', {}),
                **exp['metrics']
            }
            if exp.get('model_path'):
                row['model_path'] = exp['model_path']
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_experiments(self, metric: str, n: int = 5, ascending: bool = False) -> pd.DataFrame:
        """
        获取指定指标的最佳实验

        参数:
            metric: 指标名称
            n: 返回前n个
            ascending: 是否升序（对于R²等指标应为False）

        返回:
            包含最佳实验的DataFrame
        """
        df = self.get_history_df()
        if df.empty:
            return df

        return df.nlargest(n, metric) if not ascending else df.nsmallest(n, metric)

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        对比多个实验

        参数:
            experiment_ids: 实验ID列表

        返回:
            对比结果DataFrame
        """
        df = self.get_history_df()
        return df[df['experiment_id'].isin(experiment_ids)]

    def print_summary(self):
        """打印实验摘要"""
        df = self.get_history_df()

        if df.empty:
            print("还没有任何实验记录")
            return

        print(f"\n{'='*60}")
        print(f"实验摘要 (共{len(df)}次实验)")
        print(f"{'='*60}")

        # 按指标分组显示统计
        metric_cols = [col for col in df.columns
                      if col not in ['experiment_id', 'timestamp', 'model_path']
                      and df[col].dtype in [np.float64, np.int64]]

        if metric_cols:
            print("\n指标统计:")
            for col in metric_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    print(f"  {col}:")
                    print(f"    最佳: {values.max():.6f}")
                    print(f"    平均: {values.mean():.6f}")
                    print(f"    最差: {values.min():.6f}")

        # 显示最佳实验
        if len(metric_cols) > 0:
            primary_metric = metric_cols[0]
            best = df.nlargest(1, primary_metric).iloc[0]
            print(f"\n最佳实验 ({primary_metric}):")
            print(f"  实验ID: {best['experiment_id']}")
            print(f"  时间: {best['timestamp']}")

    def save_to_csv(self, path: Path = None):
        """将实验历史保存为CSV文件"""
        if path is None:
            path = self.experiment_dir / "experiments.csv"

        df = self.get_history_df()
        df.to_csv(path, index=False)
        print(f"实验历史已保存到: {path}")


class GridSearchRunner:
    """网格搜索运行器"""

    def __init__(self, tracker: ExperimentTracker):
        """
        初始化网格搜索运行器

        参数:
            tracker: 实验跟踪器
        """
        self.tracker = tracker

    def generate_param_grid(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        生成参数网格

        参数:
            param_grid: 参数字典，值为候选值列表

        返回:
            参数组合列表
        """
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = list(itertools.product(*values))

        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)

        return param_combinations

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        train_func: callable,
        base_params: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        运行网格搜索

        参数:
            param_grid: 要搜索的参数网格
            train_func: 训练函数，接受params参数，返回metrics
            base_params: 基础参数（固定不变的参数）

        返回:
            所有实验的结果DataFrame
        """
        param_combinations = self.generate_param_grid(param_grid)
        total = len(param_combinations)

        print(f"\n开始网格搜索: {total}组参数")
        print(f"{'='*60}")

        results = []

        for i, params in enumerate(param_combinations, 1):
            print(f"\n[{i}/{total}] 运行参数组合...")

            # 合并基础参数和搜索参数
            full_params = {**(base_params or {}), **params}

            # 开始实验
            exp_id = self.tracker.start_experiment(full_params)

            try:
                # 运行训练
                metrics = train_func(full_params)

                # 结束实验
                self.tracker.end_experiment(
                    metrics=metrics,
                    metadata={'grid_search_index': i}
                )

                results.append({
                    'experiment_id': exp_id,
                    **full_params,
                    **metrics
                })

            except Exception as e:
                print(f"实验失败: {e}")
                continue

        return pd.DataFrame(results)


# 使用示例
if __name__ == "__main__":
    # 创建实验跟踪器
    tracker = ExperimentTracker()

    # 示例1: 手动记录实验
    print("示例1: 手动记录实验")
    print("-" * 60)

    params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 500
    }

    tracker.start_experiment(params)
    tracker.log_metric('train_r2', 0.021)
    tracker.log_metric('valid_r2', 0.001)
    tracker.end_experiment(
        metrics={'train_r2': 0.021, 'valid_r2': 0.001},
        model_path='./models/model.pkl'
    )

    # 打印摘要
    tracker.print_summary()

    # 示例2: 网格搜索
    print("\n\n示例2: 网格搜索")
    print("-" * 60)

    def dummy_train_func(params):
        """模拟训练函数"""
        # 这里应该是真实的训练逻辑
        # params 包含实验参数，可以根据参数调整返回值
        import random
        # 使用params来影响结果（示例）
        lr = params.get('learning_rate', 0.05)
        depth = params.get('max_depth', 6)
        return {
            'train_r2': random.uniform(0.01, 0.03) + lr * 0.1,
            'valid_r2': random.uniform(0.0, 0.002) + depth * 0.0001
        }

    grid_search = GridSearchRunner(tracker)

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8]
    }

    grid_search.run_grid_search(
        param_grid=param_grid,
        train_func=dummy_train_func,
        base_params={'n_estimators': 500}
    )

    # 查看结果
    tracker.print_summary()
    tracker.save_to_csv()
