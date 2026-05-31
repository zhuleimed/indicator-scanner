"""
风险管理模块 (RiskManager)

职责：
  1. 涨跌停限制处理（科创板±20%、北交所±30%、主板±10%）
  2. 止盈止损策略（固定止损、百分比止盈、回落止盈）
  3. 停牌日处理
"""

from typing import Tuple

import numpy as np
import pandas as pd


class RiskManager:
    """
    风控管理器：对已生成 pos 信号的 DataFrame 施加风控规则。
    """

    # 各板块涨跌停限制
    LIMIT_RULES = {
        'kcb': (1.20, 0.80),      # 科创板 688 ±20%
        'bj': (1.30, 0.70),       # 北交所 43/83/87/88 ±30%
        'default': (1.10, 0.90),  # 主板 ±10%
    }

    def __init__(self):
        # 风控信号列
        self._stop_col = 'stop_signal'  # -1=止损, -2=止盈, -3=策略卖出

    # ------------------------------------------------------------------
    # 涨跌停限制
    # ------------------------------------------------------------------

    @staticmethod
    def get_limit_rules(stock_code: str) -> Tuple[float, float]:
        """根据股票代码确定涨跌停幅度"""
        if stock_code.startswith('688'):
            return RiskManager.LIMIT_RULES['kcb']
        elif stock_code.startswith(('430', '830', '870', '880')):
            return RiskManager.LIMIT_RULES['bj']
        return RiskManager.LIMIT_RULES['default']

    def apply_limit_up_down(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        涨停无法买入、跌停无法卖出的处理。

        逻辑：
          - 涨停开盘（open >= prev_close * up_limit）且拟买入 → pos 置 None
          - 跌停开盘（open <= prev_close * down_limit）且拟卖出 → pos 置 None
          - None 用前值前向填充
        """
        result = data.copy()
        stock_code = result['stock_code'].iloc[0]
        up_limit, down_limit = self.get_limit_rules(stock_code)

        prev_close = result['close'].shift(1)

        cannot_buy = result['open'] >= prev_close * up_limit
        result.loc[cannot_buy & (result['pos'] == 1), 'pos'] = None

        cannot_sell = result['open'] <= prev_close * down_limit
        result.loc[cannot_sell & (result['pos'] == 0), 'pos'] = None

        result['pos'] = result['pos'].ffill()
        return result

    # ------------------------------------------------------------------
    # 止盈止损
    # ------------------------------------------------------------------

    def apply_stop_strategy(self,
                            data: pd.DataFrame,
                            signal_name: str,
                            stop_loss_pct: float = 0.05,
                            stop_profit_pct: float = 0.20,
                            drawdown_pct: float = 0.03) -> pd.DataFrame:
        """
        参数化止盈止损策略。

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 pos, {signal_name}_signal, open, high, volume 列。
        signal_name : str
            信号名称，用于定位信号列 {signal_name}_signal
        stop_loss_pct : float
            止损比例（0.05 = 5%）
        stop_profit_pct : float
            止盈触发比例（0.20 = 20%）
        drawdown_pct : float
            从最高点回落止盈比例（0.03 = 3%）

        Returns
        -------
        pd.DataFrame
            新增列：stop_signal, buy_price, highest_price, position_status
        """
        result = data.copy()
        signal_col = f'{signal_name}_signal'

        # 初始化
        result[self._stop_col] = 0
        result['buy_price'] = np.nan
        result['highest_price'] = np.nan
        result['position_status'] = 0

        stop_loss = 1 - stop_loss_pct
        stop_profit = 1 + stop_profit_pct
        drawdown = 1 - drawdown_pct

        in_position = False
        curr_buy_price = np.nan
        curr_max_price = np.nan

        for i in range(len(result)):
            # 停牌日保持状态
            if result.at[i, 'volume'] == 0:
                if in_position:
                    result.at[i, 'buy_price'] = curr_buy_price
                    result.at[i, 'highest_price'] = curr_max_price
                    result.at[i, 'position_status'] = 1
                continue

            if not in_position:
                if result.at[i, signal_col] == 1:
                    curr_buy_price = result.at[i, 'open']
                    curr_max_price = curr_buy_price
                    in_position = True
                    result.at[i, 'buy_price'] = curr_buy_price
                    result.at[i, 'highest_price'] = curr_max_price
                    result.at[i, 'position_status'] = 1
            else:
                # 更新最高价
                curr_max_price = max(curr_max_price, result.at[i, 'high'])
                result.at[i, 'highest_price'] = curr_max_price
                result.at[i, 'position_status'] = 1
                open_price = result.at[i, 'open']

                # 止损
                if open_price < curr_buy_price * stop_loss:
                    result.at[i, self._stop_col] = -1
                    in_position = False
                    result.at[i, 'position_status'] = 0
                    result.at[i, 'pos'] = 0
                    continue

                # 止盈（最高价达到目标后回落）
                if curr_max_price >= curr_buy_price * stop_profit:
                    if open_price < curr_max_price * drawdown:
                        result.at[i, self._stop_col] = -2
                        in_position = False
                        result.at[i, 'position_status'] = 0
                        result.at[i, 'pos'] = 0
                        continue

                # 原始策略卖出
                if result.at[i, signal_col] == -1:
                    result.at[i, self._stop_col] = -3
                    in_position = False
                    result.at[i, 'position_status'] = 0
                    result.at[i, 'pos'] = 0

            if not in_position and result.at[i, signal_col] == 1:
                pass  # 重新买入由下一循环处理

        return result

    @staticmethod
    def is_suspended(row: pd.Series) -> bool:
        """判断当日是否停牌"""
        return row['open'] == 0 or row['volume'] == 0
