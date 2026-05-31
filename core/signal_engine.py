"""
信号引擎模块 (SignalEngine)

职责：
  1. 提供统一的信号生成接口
  2. 支持策略注册与组合（多信号合成）
  3. 将信号转换为仓位信号 (pos)
  4. 信号与策略代码完全解耦

用法:
  engine = SignalEngine()
  engine.register(strategy)                  # 注册单个策略
  engine.register(strategy2, weight=0.6)     # 可带权重
  data = engine.generate(data)              # 统一调用
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class BaseSignal:
    """
    信号策略基类。
    所有具体策略继承此类，只需实现 compute() 方法。
    """

    # 策略名称（用于列名前缀）
    name: str = 'base'
    # 默认参数字典
    params: Dict = {}

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        在 data 上添加信号列。

        约定：
          - 信号列命名为 {self.name}_signal
          - 值：1 = 买入, -1 = 卖出, 0 = 无操作
          - 可同时添加辅助列（如指标值）
        """
        raise NotImplementedError

    def __repr__(self):
        param_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'{self.__class__.__name__}({param_str})'


class SignalEngine:
    """信号生成引擎：注册 → 计算 → 合成仓位"""

    def __init__(self):
        self._strategies: list = []    # [(strategy, weight), ...]

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, strategy: BaseSignal, weight: float = 1.0):
        """注册单个策略"""
        if not isinstance(strategy, BaseSignal):
            raise TypeError(f'策略必须继承 BaseSignal，收到 {type(strategy)}')
        self._strategies.append((strategy, weight))

    def register_from(self, *strategies: BaseSignal):
        """批量注册（等权重）"""
        for s in strategies:
            self.register(s)

    def clear(self):
        """清空所有注册的策略"""
        self._strategies.clear()

    @property
    def registered(self) -> list:
        """当前已注册的策略列表"""
        return [s for s, _ in self._strategies]

    # ------------------------------------------------------------------
    # 信号生成与合成
    # ------------------------------------------------------------------

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对所有已注册策略逐一计算信号，然后合成为单一仓位列 pos。

        Returns
        -------
        pd.DataFrame
            新增列：
              - {name}_signal：每个策略的原始信号
              - pos：合成后的仓位（0 / 1）
        """
        if not self._strategies:
            raise RuntimeError('未注册任何信号策略，请先调用 register()')

        result = data.copy()

        for strategy, weight in self._strategies:
            result = strategy.compute(result)

        # 合成规则：加权求和 → 阈值判断
        total_weight = sum(w for _, w in self._strategies)
        combined = np.zeros(len(result))

        for strategy, weight in self._strategies:
            sig_col = f'{strategy.name}_signal'
            if sig_col not in result.columns:
                raise KeyError(f'策略 {strategy.name} 未生成 {sig_col} 列')
            combined += result[sig_col].fillna(0).values * (weight / total_weight)

        # 合成仓位：combined > 0 → 买入，combined < 0 → 卖出
        result['pos'] = np.where(combined > 0, 1, np.where(combined < 0, 0, np.nan))

        # 前向填充：未操作日沿用上一交易日信号
        result['pos'] = result['pos'].ffill().fillna(0)

        # 将卖出信号 pos 置 0
        result.loc[result['pos'] < 0, 'pos'] = 0

        return result

    def get_signal_columns(self, data: pd.DataFrame) -> list:
        """获取数据中所有信号列名"""
        return [c for c in data.columns if c.endswith('_signal')]
