"""
模拟盘引擎 — 015_indicator_scanner

Phase 4: 每个交易日执行：
  1. 检查交易日
  2. 执行昨日待处理订单（今日开盘价成交）
  3. 为明日生成新信号
  4. 更新持仓/现金
  5. 返回摘要供推送
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.scanner_config import (
    DATA_DIR, MAX_WORKERS,
    SLIPPAGE, COMMISSION_RATE, TAX_RATE, POSITION_PCT,
    STOP_LOSS_PCT, STOP_PROFIT_PCT, DRAWDOWN_PCT,
    INITIAL_MONEY_PER_STOCK,
)
from core.data_loader import DataLoader
from core.signal_engine import SignalEngine
from core.risk_manager import RiskManager
from core.hs300_utils import is_trading_day
from core.log_utils import get_logger
from signals.gf import GFSignal

logger = get_logger(__name__)


class Simulator:
    """Phase 4 每日模拟盘引擎"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_loader = DataLoader(data_dir)

    def run_daily(
        self,
        top_stocks: List[str],
        best_indicator: str,
        state_manager: Any,
        dry_run: bool = False,
    ) -> Optional[Dict]:
        """
        执行每日模拟盘流程。

        Parameters
        ----------
        top_stocks : list of str
            Top 10 股票代码
        best_indicator : str
            最佳指标名称
        state_manager : StateManager
            状态管理器
        dry_run : bool
            True = 只输出操作不修改状态

        Returns
        -------
        dict or None
            日报摘要，非交易日返回 None
        """
        # ---- 1. 交易日检查 ----
        if not is_trading_day():
            today_str = date.today().isoformat()
            logger.info(f'[{today_str}] 非交易日，跳过模拟盘')
            return None

        today = date.today()
        today_str = today.isoformat()
        logger.info(f'[{today_str}] 开始每日模拟盘…')

        # ---- 2. 加载今日数据 ----
        stock_data = self._load_today_data(top_stocks, best_indicator)
        if not stock_data:
            logger.warning(f'[{today_str}] 无法加载今日数据')
            return None

        # ---- 3. 从 state 获取持仓 ----
        pf = state_manager.portfolio
        cash = pf.get('cash', 1_000_000.0)
        positions = pf.get('positions', {})
        pending_orders = pf.get('pending_orders', {})
        initial_capital = pf.get('initial_capital', 1_000_000.0)

        # ---- 4. 执行昨日待处理订单 ----
        trades_today = []
        if pending_orders:
            for stock, action in list(pending_orders.items()):
                if stock not in stock_data:
                    continue
                row = stock_data[stock]['latest']

                if action == 'buy' and cash > 0:
                    trade = self._execute_buy(
                        stock, row['open'], cash, positions,
                    )
                    if trade:
                        cash = trade['cash_after']
                        trades_today.append(trade)

                elif action == 'sell' and stock in positions:
                    trade = self._execute_sell(
                        stock, row['open'], positions,
                    )
                    if trade:
                        cash += trade['net_revenue']
                        trades_today.append(trade)

            # 清空已执行的待处理订单
            pending_orders = {}

        # ---- 5. 应用风控检查（持仓是否需要止损/止盈） ----
        for stock, pos in list(positions.items()):
            if stock not in stock_data:
                continue
            row = stock_data[stock]['latest']
            if self._should_force_sell(row, pos):
                trade = self._execute_sell(stock, row['open'], positions)
                if trade:
                    cash += trade['net_revenue']
                    trade['reason'] = 'risk_stop'
                    trades_today.append(trade)

        # ---- 6. 生成明日信号 ----
        new_pending = {}
        signal_details = {}
        for stock in top_stocks:
            if stock not in stock_data:
                continue
            sig = self._get_latest_signal(
                stock_data[stock]['df'], best_indicator,
            )
            if sig == 1 and stock not in positions:
                new_pending[stock] = 'buy'
            elif sig == -1 and stock in positions:
                new_pending[stock] = 'sell'
            else:
                new_pending[stock] = 'hold'

            signal_details[stock] = {
                'signal': sig,
                'action': new_pending[stock],
            }

        # ---- 7. 计算组合摘要 ----
        portfolio_value = cash
        position_details = []
        for stock, pos in positions.items():
            if stock in stock_data:
                last_close = stock_data[stock]['latest']['close']
                market_value = pos['shares'] * last_close
                portfolio_value += market_value
                pnl_pct = (last_close - pos['avg_cost']) / pos['avg_cost']
                position_details.append({
                    'stock': stock,
                    'shares': pos['shares'],
                    'avg_cost': round(pos['avg_cost'], 2),
                    'last_close': round(last_close, 2),
                    'market_value': round(market_value, 2),
                    'pnl_pct': round(pnl_pct, 4),
                })

        daily_pnl = round(portfolio_value - (
            pf.get('_last_portfolio_value', initial_capital)
        ), 2)
        cumulative_return = round(
            (portfolio_value - initial_capital) / initial_capital, 4
        )

        # ---- 8. 更新状态 ----
        if not dry_run:
            state_manager.update_portfolio(cash, positions, new_pending)
            for trade in trades_today:
                state_manager.add_trade(trade)
            # 记录组合市值用于次日计算日收益
            pf_state = state_manager.portfolio
            pf_state['_last_portfolio_value'] = portfolio_value
            state_manager.save()

        # ---- 9. 构建摘要 ----
        summary = {
            'date': today_str,
            'indicator': best_indicator,
            'trades_today': trades_today,
            'positions': position_details,
            'cash': round(cash, 2),
            'portfolio_value': round(portfolio_value, 2),
            'initial_capital': initial_capital,
            'daily_pnl': daily_pnl,
            'cumulative_return': cumulative_return,
            'pending_orders': new_pending,
            'signal_details': signal_details,
            'dry_run': dry_run,
        }

        # 打印摘要
        self._print_summary(summary)

        return summary

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _load_today_data(
        self,
        stocks: List[str],
        indicator: str = '',
        lookback_days: int = 504,  # 约 2 年交易日
    ) -> Dict[str, Dict]:
        """
        加载每只股票的完整历史和最新一行数据。

        Returns
        -------
        dict : {stock_code: {'df': DataFrame, 'latest': Series}}
        """
        result = {}
        today = date.today()

        for code in stocks:
            try:
                # 加载足够长的历史用于指标计算
                start = today.replace(year=today.year - 3)
                df = self.data_loader.load_stock(
                    code,
                    start.strftime('%Y-%m-%d'),
                    today.strftime('%Y-%m-%d'),
                    min_days=60,
                )
                if df is not None and len(df) > 0:
                    result[code] = {
                        'df': df,
                        'latest': df.iloc[-1],
                    }
            except Exception as e:
                logger.warning(f'加载 {code} 数据失败: {e}')

        return result

    def _get_latest_signal(
        self,
        df: pd.DataFrame,
        indicator: str,
    ) -> int:
        """
        获取最新一日的信号。

        注意：信号基于最新数据的 close 计算，
        信号值为 1（买入）/ -1（卖出）/ 0（持有）。
        此信号将在下一个交易日开盘执行。

        Returns
        -------
        int : 1 (buy), -1 (sell), 0 (hold)
        """
        signal = GFSignal(indicator=indicator)
        sig_engine = SignalEngine()
        sig_engine.register(signal)

        result_df = sig_engine.generate(df.copy())

        sig_col = 'GF_signal'
        if sig_col not in result_df.columns:
            return 0

        # 获取最后一个有效信号（不 shift，因为是今天新计算的，
        # 明天开盘才执行，没有 look-ahead bias）
        return int(result_df[sig_col].iloc[-1])

    def _execute_buy(
        self,
        stock: str,
        open_price: float,
        cash: float,
        positions: Dict,
    ) -> Optional[Dict]:
        """执行买入订单。"""
        # 可用资金（按仓位比例）
        available = cash * POSITION_PCT
        if available < open_price * 100:
            return None  # 一手都买不起

        # 计算整百股
        exec_price = open_price * (1 + SLIPPAGE)
        raw_shares = int(available / exec_price)
        shares = (raw_shares // 100) * 100

        if shares == 0:
            return None

        gross_cost = shares * exec_price
        commission = max(gross_cost * COMMISSION_RATE, 5.0)
        total_cost = gross_cost + commission

        if total_cost > cash:
            return None

        # 更新持仓
        if stock in positions:
            old = positions[stock]
            new_shares = old['shares'] + shares
            new_total = old['total_cost'] + gross_cost
            positions[stock] = {
                'shares': new_shares,
                'avg_cost': round(new_total / new_shares, 4),
                'total_cost': round(new_total, 2),
            }
        else:
            positions[stock] = {
                'shares': shares,
                'avg_cost': round(exec_price, 4),
                'total_cost': round(gross_cost, 2),
            }

        return {
            'date': date.today().isoformat(),
            'stock': stock,
            'action': 'buy',
            'price': round(exec_price, 4),
            'shares': shares,
            'cost': round(total_cost, 2),
            'commission': round(commission, 2),
            'reason': 'signal_buy',
            'cash_after': round(cash - total_cost, 2),
        }

    def _execute_sell(
        self,
        stock: str,
        open_price: float,
        positions: Dict,
    ) -> Optional[Dict]:
        """执行卖出订单。"""
        if stock not in positions:
            return None

        pos = positions[stock]
        exec_price = open_price * (1 - SLIPPAGE)
        gross_revenue = pos['shares'] * exec_price
        commission = max(gross_revenue * COMMISSION_RATE, 5.0)
        tax = gross_revenue * TAX_RATE
        net_revenue = gross_revenue - commission - tax
        pnl = round(net_revenue - pos['total_cost'], 2)
        pnl_pct = round(pnl / pos['total_cost'], 4) if pos['total_cost'] > 0 else 0

        del positions[stock]

        return {
            'date': date.today().isoformat(),
            'stock': stock,
            'action': 'sell',
            'price': round(exec_price, 4),
            'shares': pos['shares'],
            'net_revenue': round(net_revenue, 2),
            'commission': round(commission, 2),
            'tax': round(tax, 2),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': 'signal_sell',
        }

    def _should_force_sell(
        self,
        row: pd.Series,
        position: Dict,
    ) -> bool:
        """
        检查持仓是否需要强制卖出（止损/止盈）。

        简化版风控——与回测逻辑保持一致：
        - 止损：开盘价 < 买入价 × (1 - stop_loss)
        """
        open_px = row.get('open', 0)
        if open_px <= 0:
            return False

        avg_cost = position.get('avg_cost', 0)
        if avg_cost <= 0:
            return False

        # 止损检查
        if open_px < avg_cost * (1 - STOP_LOSS_PCT):
            return True

        return False

    def _print_summary(self, summary: Dict):
        """打印控制台摘要。"""
        print(f'\n{"=" * 50}')
        print(f'  📊 模拟盘日报 — {summary["date"]}')
        print(f'  指标: {summary["indicator"]}')
        if summary.get('dry_run'):
            print(f'  ⚠ DRY RUN 模式（未修改状态）')
        print(f'{"=" * 50}')

        if summary['trades_today']:
            print(f'\n  当日操作:')
            for t in summary['trades_today']:
                if t['action'] == 'buy':
                    print(f'    🟢 买入 {t["stock"]}: '
                          f'{t["shares"]}股 @ {t["price"]:.2f}  '
                          f'成本={t["cost"]:.2f}')
                else:
                    print(f'    🔴 卖出 {t["stock"]}: '
                          f'{t["shares"]}股 @ {t["price"]:.2f}  '
                          f'盈亏={t.get("pnl", 0):+.2f}  '
                          f'({t.get("reason", "")})')
        else:
            print(f'\n  当日无操作')

        if summary['positions']:
            print(f'\n  持仓摘要:')
            for p in summary['positions']:
                sign = '+' if p['pnl_pct'] >= 0 else ''
                print(f'    {p["stock"]}: {p["shares"]}股  '
                      f'成本={p["avg_cost"]:.2f}  '
                      f'现价={p["last_close"]:.2f}  '
                      f'({sign}{p["pnl_pct"]:.2%})')
        else:
            print(f'\n  空仓')

        print(f'\n  账户摘要:')
        print(f'    总资产:     {summary["portfolio_value"]:,.2f}')
        print(f'    现金:       {summary["cash"]:,.2f}')
        print(f'    累计收益:   {summary["cumulative_return"]:+.2%}')
        print(f'{"=" * 50}')
