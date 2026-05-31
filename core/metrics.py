"""
绩效指标计算模块 (MetricsCalculator)

职责：
  计算回测核心绩效指标：
    - 总收益率 / 年化收益率
    - 夏普比率 / Sortino 比率
    - 最大回撤
    - 胜率（正收益日占比）
    - 交易统计（交易次数、止盈止损次数）
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .equity_curve import TradeRecord


@dataclass
class BacktestMetrics:
    """回测绩效指标容器"""
    # 收益指标
    total_return: float = 0.0          # 总收益率
    annualized_return: float = 0.0     # 年化收益率
    benchmark_return: float = 0.0      # 基准收益率
    excess_return: float = 0.0         # 超额收益

    # 风险指标
    max_drawdown: float = 0.0          # 最大回撤
    volatility: float = 0.0            # 年化波动率
    downside_volatility: float = 0.0   # 下行波动率

    # 风险调整收益
    sharpe_ratio: float = 0.0          # 夏普比率
    sortino_ratio: float = 0.0         # Sortino 比率
    calmar_ratio: float = 0.0          # Calmar 比率

    # 交易统计
    win_rate: float = 0.0              # 胜率（正收益日）
    total_trades: int = 0              # 总交易次数（买入+卖出）
    stop_loss_times: int = 0           # 止损次数
    stop_profit_times: int = 0         # 止盈次数
    avg_hold_days: float = 0.0         # 平均持股天数
    avg_profit_per_trade: float = 0.0  # 平均每笔盈利
    profit_factor: float = 0.0         # 盈亏比

    def to_dict(self) -> Dict[str, float]:
        return {
            '总收益率': round(self.total_return, 4),
            '年化收益率': round(self.annualized_return, 4),
            '基准收益率': round(self.benchmark_return, 4),
            '超额收益': round(self.excess_return, 4),
            '最大回撤': round(self.max_drawdown, 4),
            '年化波动率': round(self.volatility, 4),
            '下行波动率': round(self.downside_volatility, 4),
            '夏普比率': round(self.sharpe_ratio, 4),
            'Sortino比率': round(self.sortino_ratio, 4),
            'Calmar比率': round(self.calmar_ratio, 4),
            '胜率(日)': round(self.win_rate, 4),
            '总交易次数': self.total_trades,
            '止损次数': self.stop_loss_times,
            '止盈次数': self.stop_profit_times,
            '平均持股天数': round(self.avg_hold_days, 1),
            '平均每笔盈利': round(self.avg_profit_per_trade, 2),
            '盈亏比': round(self.profit_factor, 4),
        }


class MetricsCalculator:
    """绩效指标计算器"""

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, risk_free_rate: float = 0.027):
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute(self,
                account_data: pd.DataFrame,
                initial_money: float,
                trade_records: List[List[TradeRecord]] = None,
                benchmark_return: float = None) -> BacktestMetrics:
        """
        计算完整回测绩效指标。

        Parameters
        ----------
        account_data : pd.DataFrame
            账户级资金曲线（含 equity, equity_returns, equity_cumulative_returns）
        initial_money : float
            初始资金
        trade_records : list of lists, optional
            每只股票的交易记录
        benchmark_return : float, optional
            基准总收益率
        """
        metrics = BacktestMetrics()

        # 收益指标
        final_equity = account_data['equity'].iloc[-1]
        metrics.total_return = (final_equity - initial_money) / initial_money

        n = len(account_data)
        final_cum = account_data['equity_cumulative_returns'].iloc[-1]
        metrics.annualized_return = (final_cum ** (self.TRADING_DAYS_PER_YEAR / n) - 1)

        if benchmark_return is not None:
            metrics.benchmark_return = benchmark_return
            metrics.excess_return = metrics.total_return - benchmark_return

        # 风险指标
        daily_returns = account_data['equity_returns'].dropna()
        metrics.volatility = daily_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        cumulative = account_data['equity_cumulative_returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = drawdown.min()

        # 下行波动率
        rf_daily = self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
        downside = daily_returns[daily_returns < rf_daily]
        metrics.downside_volatility = (
            downside.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            if len(downside) > 0 else 0.0
        )

        # 风险调整收益
        avg_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        if std_daily > 0:
            metrics.sharpe_ratio = (
                (avg_daily - rf_daily) / std_daily
                * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            )
        if metrics.downside_volatility > 0:
            metrics.sortino_ratio = (
                (avg_daily - rf_daily) / (downside.std() if len(downside) > 0 else 1)
                * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            )
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

        # 交易统计
        metrics.win_rate = (daily_returns > 0).sum() / len(daily_returns)

        if trade_records:
            self._compute_trade_stats(metrics, trade_records)

        return metrics

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trade_stats(metrics: BacktestMetrics,
                              trade_records: List[List[TradeRecord]]):
        """计算交易统计指标"""
        all_trades = [t for sub in trade_records for t in sub]
        metrics.total_trades = len(all_trades)

        sell_trades = [t for t in all_trades if '卖出' in t.trade_type]
        metrics.stop_loss_times = sum(
            1 for t in sell_trades if t.trade_type == '止损卖出'
        )
        metrics.stop_profit_times = sum(
            1 for t in sell_trades if t.trade_type == '止盈卖出'
        )

        # 平均持股天数（仅卖出交易）
        if sell_trades:
            metrics.avg_hold_days = np.mean([t.days_held for t in sell_trades])

        # 平均每笔盈利和盈亏比
        profitable_trades = [t for t in sell_trades if t.profit > 0]
        losing_trades = [t for t in sell_trades if t.profit < 0]

        if sell_trades:
            metrics.avg_profit_per_trade = np.mean([t.profit for t in sell_trades])

        total_profit = sum(t.profit for t in profitable_trades)
        total_loss = abs(sum(t.profit for t in losing_trades))
        if total_loss > 0:
            metrics.profit_factor = total_profit / total_loss
