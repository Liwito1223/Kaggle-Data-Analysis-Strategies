#!/usr/bin/env python3
"""
实验运行脚本

使用方法:
1. 运行单个预设实验:
   python run_experiments.py --experiment standard

2. 运行网格搜索:
   python run_experiments.py --grid_search --param learning_rate,max_depth

3. 运行所有预设实验:
   python run_experiments.py --all

4. 列出可用配置:
   python run_experiments.py --list
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from experiment_tracker import ExperimentTracker, GridSearchRunner
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import gc


def load_config(config_file: str = "experiment_config.yaml") -> Dict:
    """加载配置文件"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def weighted_r2_score(y_true, y_pred, sample_weight):
    """计算加权R²得分"""
    numerator = np.average((y_true - y_pred) ** 2, weights=sample_weight)
    denominator = np.average(y_true ** 2, weights=sample_weight) + 1e-38
    r2 = 1 - numerator / denominator
    return r2


def lgb_r2_metric(y_true, y_pred, sample_weight):
    """LightGBM格式的加权R²评估函数"""
    r2 = weighted_r2_score(y_true, y_pred, sample_weight)
    return 'r2', r2, True


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config_file: str = "experiment_config.yaml"):
        """初始化实验运行器"""
        self.config = load_config(config_file)
        self.tracker = ExperimentTracker()

        # 特征列
        self.feature_cols = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{i}_lag_1" for i in range(9)]

    def _build_params(self, experiment_name: str) -> Dict[str, Any]:
        """
        根据实验名称构建完整的参数字典

        参数:
            experiment_name: 实验名称

        返回:
            完整的参数字典
        """
        base = self.config.get('base_config', {})
        exp_config = self.config.get('experiments', {}).get(experiment_name, {})

        # 合并配置
        params = {}

        # 数据配置
        data_config = {**base.get('data', {}), **exp_config}
        params.update(data_config)

        # 特征配置
        params['feature_cols'] = self.feature_cols
        params['target_col'] = base.get('features', {}).get('target_col', 'responder_6')
        params['weight_col'] = base.get('features', {}).get('weight_col', 'weight')

        # 训练配置
        training_config = {**base.get('training', {})}
        params.update(training_config)

        # 系统配置
        system_config = base.get('system', {})
        params['device'] = system_config.get('device', 'cpu')
        params['model_output_dir'] = system_config.get('model_output_dir', './models')

        return params

    def run_single_experiment(self, experiment_name: str) -> Dict[str, float]:
        """
        运行单个实验

        参数:
            experiment_name: 实验名称

        返回:
            评估指标字典
        """
        print(f"\n运行实验: {experiment_name}")
        print("="*60)

        # 构建参数
        params = self._build_params(experiment_name)

        # 提取数据参数
        n_train_dates = params.get('n_train_dates', 15)
        n_valid_dates = params.get('n_valid_dates', 15)
        n_symbols = params.get('n_symbols', 20)
        train_file = params.get('train_file', './processed_data/training.parquet')
        valid_file = params.get('valid_file', './processed_data/validation.parquet')

        # 准备需要的列
        required_cols = (self.feature_cols +
                        [params['target_col'], params['weight_col'],
                         'date_id', 'symbol_id'])

        # 加载数据（直接使用polars，精确控制采样）
        print(f"\n加载数据配置:")
        print(f"  训练集: {n_train_dates}天")
        print(f"  验证集: {n_valid_dates}天")
        print(f"  股票数: {n_symbols}个")

        import polars as pl

        # ===== 加载训练数据 =====
        print("\n[1/2] 加载训练数据...")
        train_lazy = pl.scan_parquet(train_file)

        # 检查哪些列存在
        schema = train_lazy.collect_schema()
        available_cols = [col for col in required_cols if col in schema.names()]

        # 只保留需要的列
        train_lazy = train_lazy.select(available_cols)

        # 获取所有日期并选择最后n天
        all_dates = train_lazy.select('date_id').unique().collect().to_series().to_list()
        all_dates.sort()
        selected_train_dates = all_dates[-n_train_dates:]

        # 获取所有股票并选择前n个
        all_symbols = train_lazy.select('symbol_id').unique().collect().to_series().to_list()
        all_symbols.sort()
        selected_symbols = all_symbols[:n_symbols]

        print(f"  选择日期: {selected_train_dates[0]}-{selected_train_dates[-1]} ({len(selected_train_dates)}天)")
        print(f"  选择股票: {len(selected_symbols)}个")

        # 过滤数据
        train_lazy = train_lazy.filter(
            pl.col('date_id').is_in(selected_train_dates) &
            pl.col('symbol_id').is_in(selected_symbols)
        )

        # 填充缺失值
        for col in self.feature_cols:
            if col in available_cols:
                train_lazy = train_lazy.with_columns(
                    pl.col(col).fill_null(0)
                )

        # 转换为pandas
        train_pl = train_lazy.collect()
        train_df = train_pl.to_pandas()

        print(f"  ✓ 训练数据: {train_df.shape}")

        del train_lazy, train_pl
        gc.collect()

        # ===== 加载验证数据 =====
        print("\n[2/2] 加载验证数据...")

        if Path(valid_file).exists():
            valid_lazy = pl.scan_parquet(valid_file)

            # 检查列
            schema = valid_lazy.collect_schema()
            available_cols = [col for col in required_cols if col in schema.names()]
            valid_lazy = valid_lazy.select(available_cols)

            # 选择日期和股票
            all_dates = valid_lazy.select('date_id').unique().collect().to_series().to_list()
            all_dates.sort()
            selected_valid_dates = all_dates[-n_valid_dates:]

            valid_lazy = valid_lazy.filter(
                pl.col('date_id').is_in(selected_valid_dates) &
                pl.col('symbol_id').is_in(selected_symbols)
            )

            # 填充缺失值
            for col in self.feature_cols:
                if col in available_cols:
                    valid_lazy = valid_lazy.with_columns(
                        pl.col(col).fill_null(0)
                    )

            valid_pl = valid_lazy.collect()
            valid_df = valid_pl.to_pandas()

            print(f"  ✓ 验证数据: {valid_df.shape}")

            del valid_lazy, valid_pl
            gc.collect()
        else:
            print("  验证数据文件不存在")
            valid_df = None

        # 准备训练数据
        X_train = train_df[self.feature_cols]
        y_train = train_df[params['target_col']]
        w_train = train_df[params['weight_col']]

        X_valid = valid_df[self.feature_cols]
        y_valid = valid_df[params['target_col']]
        w_valid = valid_df[params['weight_col']]

        # 清理原始数据
        del train_df, valid_df
        gc.collect()

        # 创建LightGBM参数
        lgb_params = {
            'n_estimators': params.get('n_estimators', 500),
            'learning_rate': params.get('learning_rate', 0.05),
            'max_depth': params.get('max_depth', 6),
            'num_leaves': params.get('num_leaves', 64),
            'min_data_in_leaf': params.get('min_data_in_leaf', 500),
            'objective': 'regression',
            'metric': 'None',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'device': params['device'],
        }

        print(f"\nLightGBM参数:")
        for key, value in lgb_params.items():
            print(f"  {key}: {value}")

        # 训练模型
        print("\n开始训练...")
        model = lgb.LGBMRegressor(**lgb_params)

        model.fit(
            X_train, y_train, w_train,
            eval_set=[(X_valid, y_valid, w_valid)],
            eval_metric=lgb_r2_metric,
            callbacks=[
                lgb.early_stopping(params.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(100)
            ]
        )

        # 评估
        train_r2 = weighted_r2_score(y_train, model.predict(X_train), w_train)
        valid_r2 = weighted_r2_score(y_valid, model.predict(X_valid), w_valid)

        metrics = {
            'train_r2': train_r2,
            'valid_r2': valid_r2
        }

        # 保存模型
        model_dir = Path(params['model_output_dir'])
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / f"model_{experiment_name}.pkl"
        joblib.dump(model, model_path)

        print(f"\n模型已保存: {model_path}")

        # 清理
        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

        return metrics

    def run_grid_search(
        self,
        param_names: str,
        base_experiment: str = "standard"
    ) -> pd.DataFrame:
        """
        运行网格搜索

        参数:
            param_names: 要搜索的参数名，逗号分隔，如 "learning_rate,max_depth"
            base_experiment: 基础实验名称

        返回:
            结果DataFrame
        """
        # 获取基础参数
        base_params = self._build_params(base_experiment)

        # 解析要搜索的参数
        param_list = [p.strip() for p in param_names.split(',')]

        # 从配置中获取参数网格
        grid_config = self.config.get('grid_search', {}).get('lgbm_params', {})

        param_grid = {}
        for param_name in param_list:
            if param_name in grid_config:
                param_grid[param_name] = grid_config[param_name]
            else:
                print(f"警告: 参数 '{param_name}' 在配置文件中未定义，跳过")

        if not param_grid:
            print("错误: 没有找到要搜索的参数")
            return pd.DataFrame()

        print(f"\n网格搜索配置:")
        print(f"  搜索参数: {list(param_grid.keys())}")
        print(f"  总组合数: {np.prod([len(v) for v in param_grid.values()])}")

        # 创建训练函数
        def train_func(params):
            return self.run_single_experiment('temp')

        # 运行网格搜索
        grid_runner = GridSearchRunner(self.tracker)
        results = grid_runner.run_grid_search(
            param_grid=param_grid,
            train_func=train_func,
            base_params=base_params
        )

        return results

    def list_experiments(self):
        """列出所有可用的实验"""
        print("\n可用的实验配置:")
        print("="*60)

        experiments = self.config.get('experiments', {})
        for name, config in experiments.items():
            print(f"\n{name}:")
            print(f"  名称: {config.get('name', name)}")
            print(f"  描述: ", end="")
            for key, value in config.items():
                if key != 'name':
                    print(f"{key}={value} ", end="")
            print()

    def run_all_experiments(self):
        """运行所有预设实验"""
        experiments = self.config.get('experiments', {})

        print(f"\n将运行 {len(experiments)} 个实验:")
        for name in experiments.keys():
            print(f"  - {name}")

        results = {}

        for exp_name in experiments.keys():
            try:
                metrics = self.run_single_experiment(exp_name)
                results[exp_name] = metrics
            except Exception as e:
                print(f"实验 '{exp_name}' 失败: {e}")
                results[exp_name] = None

        # 打印总结
        print("\n" + "="*60)
        print("所有实验完成!")
        print("="*60)

        for exp_name, metrics in results.items():
            if metrics:
                print(f"\n{exp_name}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.6f}")

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行机器学习实验')

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='运行单个实验（实验名称）'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='运行所有预设实验'
    )

    parser.add_argument(
        '--grid_search', '-g',
        action='store_true',
        help='运行网格搜索'
    )

    parser.add_argument(
        '--param', '-p',
        type=str,
        help='网格搜索的参数（逗号分隔）'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用实验'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='experiment_config.yaml',
        help='配置文件路径'
    )

    args = parser.parse_args()

    # 创建运行器
    runner = ExperimentRunner(args.config)

    # 执行相应操作
    if args.list:
        runner.list_experiments()

    elif args.all:
        runner.run_all_experiments()

    elif args.grid_search:
        if not args.param:
            print("错误: 网格搜索需要指定参数 (--param)")
            parser.print_help()
            return

        runner.run_grid_search(args.param)

    elif args.experiment:
        runner.run_single_experiment(args.experiment)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
