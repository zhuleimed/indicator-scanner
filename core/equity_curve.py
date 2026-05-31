"""
资金曲线计算模块 (EquityCurveCalculator)

职责：
  1. 单只股票的资金曲线计算（含滑点、佣金、印花税）
  2. 多只股票的账户聚合资金曲线
  3. 交易明细记录（买入/卖出/止盈止损/虚拟卖出）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """单笔交易记录"""
    date: str
    trade_type: str          # 买入 / 卖出 / 止损卖出 / 止盈卖出 / 策略卖出 / 虚拟卖出
    price: float
    volume: int               # 成交股数
    commission: float         # 手续费
    tax: float                # 印花税
    total_cost: float         # 总成本
    total_revenue: float      # 总收入
    profit: float             # 盈亏金额
    days_held: int            # 持股天数
    return_rate: float        # 收益率


@dataclass
class BacktestParams:
    """回测交易参数"""
    initial_money: float = 10_000.0
    slippage: float = 0.003        # 滑点 0.3%
    commission_rate: float = 0.0005  # 佣金万分之五
    tax_rate: float = 0.001         # 印花税千分之一
    position_pct: float = 0.95      # 仓位比例 95%


class EquityCurveCalculator:
    """资金曲线计算器"""

    # 交易类型映射
    TRADE_TYPE_MAP = {-1: '止损卖出', -2: '止盈卖出', -3: '策略卖出'}

    def __init__(self, params: Optional[BacktestParams] = None):
        self.params = params or BacktestParams()

    # ------------------------------------------------------------------
    # 单只股票资金曲线
    # ------------------------------------------------------------------

    def compute_single(self, data: pd.DataFrame) -> Dict:
        """
        根据 pos 信号计算单只股票的资金曲线与交易记录。

        Parameters
        ----------
        data : pd.DataFrame
            必须包含：date, open, close, volume, pos, stop_signal

        Returns
        -------
        dict
            {'equity_curve': DataFrame, 'trade_records': List[TradeRecord],
             'open_buys': List 未平仓}
        """
        df = data.copy().reset_index(drop=True)

        # 初始化列
        for col in ['hold_num', 'stock_value', 'cash', 'equity', '手续费', '印花税']:
            df[col] = 0.0
        df.loc[0, 'cash'] = self.params.initial_money
        df.loc[0, 'equity'] = self.params.initial_money

        trade_records: List[TradeRecord] = []
        open_buys: List[Dict] = []  # 跟踪未平仓买入订单

        for i in range(1, len(df)):
            # 停牌日处理
            if df.at[i, 'open'] == 0 or df.at[i, 'volume'] == 0:
                df.at[i, 'hold_num'] = df.at[i - 1, 'hold_num']
                df.at[i, 'cash'] = df.at[i - 1, 'cash']
                df.at[i, 'stock_value'] = df.at[i, 'hold_num'] * df.at[i, 'close']
                df.at[i, 'equity'] = df.at[i, 'cash'] + df.at[i, 'stock_value']
                continue

            prev_hold = df.at[i - 1, 'hold_num']
            pos_change = df.at[i, 'pos'] != df.at[i - 1, 'pos']

            if pos_change:
                theory_num = (df.at[i - 1, 'equity']
                              * df.at[i, 'pos']
                              * self.params.position_pct
                              / df.at[i, 'open'])
                theory_num = int(theory_num // 100 * 100)  # 整百股

                if theory_num >= prev_hold:
                    # ---- 买入 ----
                    buy_num = theory_num - prev_hold
                    self._execute_buy(df, i, buy_num, open_buys, trade_records)
                else:
                    # ---- 卖出 ----
                    sell_num = prev_hold - theory_num
                    stop_signal = df.at[i, 'stop_signal']
                    self._execute_sell(df, i, sell_num, stop_signal,
                                       open_buys, trade_records)
            else:
                df.at[i, 'hold_num'] = prev_hold
                df.at[i, 'cash'] = df.at[i - 1, 'cash']

            # 更新市值和总资产
            df.at[i, 'stock_value'] = df.at[i, 'hold_num'] * df.at[i, 'close']
            df.at[i, 'equity'] = df.at[i, 'cash'] + df.at[i, 'stock_value']

        # 未平仓 → 虚拟卖出
        self._close_open_positions(df, open_buys, trade_records)

        # 计算资金曲线收益率
        df['equity_returns'] = df['equity'].pct_change()
        df['equity_cumulative_returns'] = (1 + df['equity_returns']).cumprod()
        df.loc[0, 'equity_cumulative_returns'] = 1.0

        return {'equity_curve': df, 'trade_records': trade_records,
                'open_buys': open_buys}

    # ------------------------------------------------------------------
    # 账户聚合资金曲线
    # ------------------------------------------------------------------

    def compute_account(self, stock_results: List[Dict]) -> pd.DataFrame:
        """
        将多只股票的资金曲线聚合为账户级别。

        Parameters
        ----------
        stock_results : list of dict
            每个元素是 compute_single 的返回值

        Returns
        -------
        pd.DataFrame
            账户级资金曲线
        """
        curves = [r['equity_curve'] for r in stock_results]

        all_dates = sorted(set(
            d for c in curves for d in c['date'].dt.strftime('%Y-%m-%d')
        ))
        account = pd.DataFrame({'date': pd.to_datetime(all_dates)})
        account['cash'] = 0.0
        account['stock_value'] = 0.0
        account['手续费'] = 0.0
        account['印花税'] = 0.0
        account['equity'] = 0.0

        for c in curves:
            for _, row in c.iterrows():
                mask = account['date'] == row['date']
                if mask.any():
                    account.loc[mask, 'cash'] += row['cash']
                    account.loc[mask, 'stock_value'] += row['stock_value']
                    account.loc[mask, '手续费'] += row['手续费']
                    account.loc[mask, '印花税'] += row['印花税']
                    account.loc[mask, 'equity'] += row['equity']

        account['equity_returns'] = account['equity'].pct_change().fillna(0)
        account['equity_cumulative_returns'] = (
            1 + account['equity_returns']
        ).cumprod()
        account.loc[0, 'equity_cumulative_returns'] = 1.0
        return account

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _execute_buy(self, df, i, buy_num, open_buys, trade_records):
        """执行买入"""
        if buy_num <= 0:
            # 资金不足或价格过高导致可买数量为 0，保持现状
            df.at[i, 'hold_num'] = df.at[i - 1, 'hold_num']
            df.at[i, 'cash'] = df.at[i - 1, 'cash']
            return
        buy_price = df.at[i, 'open'] * (1 + self.params.slippage)
        buy_cost = buy_num * buy_price
        commission = max(round(buy_cost * self.params.commission_rate, 2), 5.0)
        total_cost = buy_cost + commission

        open_buys.append({
            '数量': buy_num, '价格': buy_price, '手续费': commission,
            '总成本': total_cost, '日期': df.at[i, 'date'],
        })

        trade_records.append(TradeRecord(
            date=str(df.at[i, 'date'].date()),
            trade_type='买入',
            price=buy_price,
            volume=buy_num,
            commission=commission,
            tax=0.0,
            total_cost=total_cost,
            total_revenue=0.0,
            profit=0.0,
            days_held=0,
            return_rate=0.0,
        ))

        df.at[i, 'hold_num'] = df.at[i - 1, 'hold_num'] + buy_num
        df.at[i, 'cash'] = df.at[i - 1, 'cash'] - total_cost

    def _execute_sell(self, df, i, sell_num, stop_signal, open_buys, trade_records):
        """执行卖出"""
        if sell_num <= 0:
            return
        # 清理队列中可能存在的异常买单（数量为0）
        open_buys[:] = [o for o in open_buys if o.get('数量', 0) > 0]

        sell_price = df.at[i, 'open'] * (1 - self.params.slippage)
        sell_revenue = sell_num * sell_price
        commission = max(round(sell_revenue * self.params.commission_rate, 2), 5.0)
        tax = round(sell_revenue * self.params.tax_rate, 2)
        total_revenue = sell_revenue - commission - tax

        trade_type = self.TRADE_TYPE_MAP.get(stop_signal, '卖出')
        remaining = sell_num

        while remaining > 0 and open_buys:
            buy_order = open_buys[0]
            batch = min(buy_order['数量'], remaining)
            batch_cost = buy_order['总成本'] * (batch / buy_order['数量'])
            # 佣金和印花税按卖出数量比例分摊，避免重复扣减
            batch_commission = commission * (batch / sell_num)
            batch_tax = tax * (batch / sell_num)
            profit = (sell_price - buy_order['价格']) * batch \
                     - buy_order['手续费'] * (batch / buy_order['数量']) \
                     - batch_commission - batch_tax
            days_held = (df.at[i, 'date'] - buy_order['日期']).days
            return_rate = profit / batch_cost if batch_cost != 0 else 0.0

            trade_records.append(TradeRecord(
                date=str(df.at[i, 'date'].date()),
                trade_type=trade_type,
                price=sell_price,
                volume=batch,
                commission=batch_commission,
                tax=batch_tax,
                total_cost=batch_cost,
                total_revenue=total_revenue * (batch / sell_num),
                profit=profit,
                days_held=days_held,
                return_rate=return_rate,
            ))

            buy_order['数量'] -= batch
            if buy_order['数量'] <= 0:
                open_buys.pop(0)
            else:
                buy_order['总成本'] -= batch_cost
            remaining -= batch

        df.at[i, 'hold_num'] = df.at[i - 1, 'hold_num'] - sell_num
        df.at[i, 'cash'] = df.at[i - 1, 'cash'] + total_revenue

    def _close_open_positions(self, df, open_buys, trade_records):
        """期末未平仓 → 虚拟卖出"""
        if not open_buys:
            return
        final_date = df['date'].iloc[-1]
        final_close = df['close'].iloc[-1]

        for order in open_buys:
            days_held = (final_date - order['日期']).days
            current_value = order['数量'] * final_close
            profit = current_value - order['总成本']
            return_rate = profit / order['总成本'] if order['总成本'] != 0 else 0.0

            trade_records.append(TradeRecord(
                date=str(final_date.date()),
                trade_type='虚拟卖出',
                price=final_close,
                volume=order['数量'],
                commission=0.0,
                tax=0.0,
                total_cost=order['总成本'],
                total_revenue=current_value,
                profit=profit,
                days_held=days_held,
                return_rate=return_rate,
            ))
