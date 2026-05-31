"""
扫描引擎 — 015_indicator_scanner

Phase 1: 全指标扫描 → 找到最稳健的普适性指标
Phase 2: 用最佳指标选 Top N 股票
Phase 3: 验证选股在近 3 个月是否仍有效

核心设计：
  - 轻量回测：不写文件、不画图、不打印日志
  - 多进程并行：ProcessPoolExecutor，真正利用多核 CPU
  - 按指标分批：Phase 1 提交 97 个任务（每个处理 ~300 只股票）
  - worker 自行加载数据，避免 pickle 传输 DataFrame 的开销
"""

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.scanner_config import (
    SCAN_YEARS, VERIFY_MONTHS, TOP_N_STOCKS,
    MAX_WORKERS, STD_PENALTY, MIN_TRADING_DAYS,
    STOP_LOSS_PCT, STOP_PROFIT_PCT, DRAWDOWN_PCT,
    INITIAL_MONEY_PER_STOCK, SLIPPAGE, COMMISSION_RATE,
    TAX_RATE, POSITION_PCT, BENCHMARK_CODE,
    VERIFY_PASS_THRESHOLD,
)

from core.data_loader import DataLoader
from core.signal_engine import SignalEngine
from core.risk_manager import RiskManager
from core.equity_curve import BacktestParams, EquityCurveCalculator
from core.log_utils import get_logger
from core.hs300_utils import load_index_data
from signals.gf import GFSignal

logger = get_logger(__name__)


# ======================================================================
# 模块级 worker 函数 — ProcessPoolExecutor 要求可 pickle 的函数
# ======================================================================

def _worker_run_backtest(
    stock_df: pd.DataFrame,
    indicator: str,
    benchmark_return: float,
) -> float:
    """
    核心计算单元：对单只股票+单个指标执行轻量回测，返回超额收益率。

    此函数必须定义在模块级别，以便 ProcessPoolExecutor 序列化。
    所有依赖在函数内部导入，避免主进程的状态泄漏到子进程。
    """
    # ---- 1. 信号生成 ----
    signal = GFSignal(indicator=indicator)
    sig_engine = SignalEngine()
    sig_engine.register(signal)
    df = sig_engine.generate(stock_df)

    # ---- 2. Look-ahead bias 修正 ----
    sig_col = 'GF_signal'
    if sig_col in df.columns:
        df[sig_col] = df[sig_col].shift(1).fillna(0)
    df['pos'] = df['pos'].shift(1).ffill().fillna(0)

    # ---- 3. 风控 ----
    risk = RiskManager()
    df = risk.apply_limit_up_down(df)
    df = risk.apply_stop_strategy(
        df, 'GF',
        stop_loss_pct=STOP_LOSS_PCT,
        stop_profit_pct=STOP_PROFIT_PCT,
        drawdown_pct=DRAWDOWN_PCT,
    )

    # 信号修正
    for i in range(1, len(df)):
        if df.at[i, sig_col] == -1 and df.at[i, 'stop_signal'] in [-1, -2, -3]:
            df.at[i, 'pos'] = 0
    df['pos'] = df['pos'].ffill().fillna(0)

    # ---- 4. 资金曲线 ----
    params = BacktestParams(
        initial_money=INITIAL_MONEY_PER_STOCK,
        slippage=SLIPPAGE,
        commission_rate=COMMISSION_RATE,
        tax_rate=TAX_RATE,
        position_pct=POSITION_PCT,
    )
    eq_calc = EquityCurveCalculator(params)
    result = eq_calc.compute_single(df)

    # ---- 5. 策略总收益 ----
    equity = result['equity_curve']['equity']
    final_equity = equity.iloc[-1]
    strategy_return = (
        final_equity - INITIAL_MONEY_PER_STOCK
    ) / INITIAL_MONEY_PER_STOCK

    return strategy_return - benchmark_return


def _worker_scan_indicator(args: Tuple) -> Tuple[str, List[float]]:
    """
    Phase 1 worker：加载数据并对一个指标扫描全部股票。

    Parameters
    ----------
    args : tuple
        (indicator, stock_codes, start_date, end_date,
         data_dir, benchmark_return)

    Returns
    -------
    (indicator, [excess_return, ...])
    """
    (indicator, stock_codes, start_date, end_date,
     data_dir, benchmark_return) = args

    # 每个子进程内独立创建 DataLoader
    loader = DataLoader(data_dir if data_dir else None)
    excess_returns = []

    for code in stock_codes:
        try:
            df = loader.load_stock(
                code, start_date, end_date, min_days=MIN_TRADING_DAYS,
            )
        except Exception:
            continue

        if df is None or len(df) < 60:
            continue

        try:
            excess = _worker_run_backtest(df, indicator, benchmark_return)
            excess_returns.append(excess)
        except Exception:
            continue

    return (indicator, excess_returns)


def _worker_select_stocks(args: Tuple) -> List[Dict]:
    """
    Phase 2 worker：对一批股票用最佳指标回测。

    Parameters
    ----------
    args : tuple
        (indicator, stock_codes, start_date, end_date,
         data_dir, benchmark_return)

    Returns
    -------
    [{'stock': code, 'excess_return': float}, ...]
    """
    (indicator, stock_codes, start_date, end_date,
     data_dir, benchmark_return) = args

    loader = DataLoader(data_dir if data_dir else None)
    results = []

    for code in stock_codes:
        try:
            df = loader.load_stock(
                code, start_date, end_date, min_days=MIN_TRADING_DAYS,
            )
        except Exception:
            continue

        if df is None or len(df) < 60:
            continue

        try:
            excess = _worker_run_backtest(df, indicator, benchmark_return)
            results.append({'stock': code, 'excess_return': round(float(excess), 6)})
        except Exception:
            continue

    return results


# ======================================================================
# ScannerEngine
# ======================================================================

class ScannerEngine:
    """Phase 1-3 扫描编排器"""

    def __init__(self, data_dir: str = None):
        self.data_loader = DataLoader(data_dir)
        self._data_dir = data_dir  # 传给子进程的 DataLoader

    # ==================================================================
    # Phase 1: 全指标扫描
    # ==================================================================

    def scan_all_indicators(
        self,
        stock_codes: List[str],
        start_date: str = '',
        end_date: str = '',
    ) -> List[Dict]:
        """扫描全部 97 个指标，找出最稳健的普适性指标。"""
        indicators = list(GFSignal.INDICATORS)

        if not start_date:
            end_dt = date.today()
            start_dt = end_dt.replace(year=end_dt.year - SCAN_YEARS)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        # 处理 end_date 超出数据范围的问题
        effective_end = end_date
        if effective_end:
            try:
                end_dt = date.fromisoformat(effective_end)
                if end_dt >= date.today():
                    effective_end = ''
            except (ValueError, TypeError):
                pass

        print(f'\n{"=" * 60}')
        print(f'[Phase 1] 全指标扫描')
        print(f'  指标数: {len(indicators)}')
        print(f'  股票数: {len(stock_codes)}')
        print(f'  回测期: {start_date} → {end_date}')
        print(f'  并行:   {MAX_WORKERS} 进程 (ProcessPoolExecutor)')
        print(f'{"=" * 60}')

        t0 = time.time()

        # ---- 快速预扫描：确定有效股票 ----
        print('\n[1/3] 预检查有效股票…')
        valid_codes = []
        for code in stock_codes:
            try:
                df = self.data_loader.load_stock(
                    code, start_date, effective_end, min_days=MIN_TRADING_DAYS,
                )
                if df is not None and len(df) >= 60:
                    valid_codes.append(code)
            except Exception:
                pass
        print(f'  有效股票: {len(valid_codes)}/{len(stock_codes)}')

        if len(valid_codes) < 50:
            print('  ⚠ 有效股票过少，扫描结果可能不可靠')

        # ---- 计算基准收益 ----
        print('\n[2/3] 加载基准指数…')
        benchmark_return = self._get_benchmark_return(start_date, end_date)
        print(f'  基准（沪深300）收益率: {benchmark_return:.2%}')

        # ---- 多进程并行扫描（按指标分批）----
        print(f'\n[3/3] 并行扫描 {len(indicators)} 个指标…')
        n_indicators = len(indicators)

        # 构建任务：每个指标一个进程，进程内自行加载所有股票数据
        tasks = [
            (ind, valid_codes, start_date, effective_end,
             self._data_dir, benchmark_return)
            for ind in indicators
        ]

        indicator_results: Dict[str, List[float]] = defaultdict(list)
        completed = 0

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_worker_scan_indicator, task): task[0]
                for task in tasks
            }

            for future in as_completed(futures):
                indicator_name = futures[future]
                completed += 1
                try:
                    ind, excess_list = future.result()
                    if excess_list:
                        indicator_results[ind] = excess_list
                except Exception as e:
                    logger.warning(f'指标 {indicator_name} 扫描失败: {e}')

                # 进度报告
                pct = completed / n_indicators * 100
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_indicators - completed) / rate if rate > 0 else 0
                print(f'  进度: {completed}/{n_indicators} 指标 '
                      f'({pct:.0f}%)  ETA: {eta:.0f}s', end='\r')

        elapsed = time.time() - t0
        print(f'\n  完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)')

        # ---- 计算综合评分 ----
        print('\n计算综合评分…')
        scores = []
        for indicator in indicators:
            excess_list = indicator_results.get(indicator, [])
            num_valid = len(excess_list)
            if num_valid < 10:
                continue

            arr = np.array(excess_list)
            mean_excess = float(np.mean(arr))
            win_rate = float(np.mean(arr > 0))
            std_excess = float(np.std(arr))
            score = mean_excess * win_rate - STD_PENALTY * std_excess

            scores.append({
                'indicator': indicator,
                'score': round(score, 6),
                'mean_excess': round(mean_excess, 6),
                'win_rate': round(win_rate, 4),
                'std_excess': round(std_excess, 6),
                'num_valid': num_valid,
            })

        scores.sort(key=lambda x: x['score'], reverse=True)

        # 打印 Top 10
        self._print_top_scores(scores)

        if not scores:
            raise RuntimeError('扫描失败：所有指标的有效样本均不足')

        return scores

    # ==================================================================
    # Phase 2: 选股
    # ==================================================================

    def select_top_stocks(
        self,
        stock_codes: List[str],
        best_indicator: str,
        start_date: str = '',
        end_date: str = '',
    ) -> Tuple[List[str], List[Dict]]:
        """用最佳指标对全部股票回测，选出超额收益最高的 Top N 股票。"""
        if not start_date:
            end_dt = date.today()
            start_dt = end_dt.replace(year=end_dt.year - SCAN_YEARS)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        effective_end = end_date
        if effective_end:
            try:
                if date.fromisoformat(effective_end) >= date.today():
                    effective_end = ''
            except (ValueError, TypeError):
                pass

        # 预检查有效股票
        valid_codes = []
        for code in stock_codes:
            try:
                df = self.data_loader.load_stock(
                    code, start_date, effective_end, min_days=MIN_TRADING_DAYS,
                )
                if df is not None and len(df) >= 60:
                    valid_codes.append(code)
            except Exception:
                pass

        print(f'\n{"=" * 60}')
        print(f'[Phase 2] 选股 — 指标: {best_indicator}')
        print(f'  股票数: {len(valid_codes)} (有效)')
        print(f'  并行:   {MAX_WORKERS} 进程')
        print(f'{"=" * 60}')

        benchmark_return = self._get_benchmark_return(start_date, end_date)

        # 将股票分成 MAX_WORKERS 个块，每个进程处理一块
        chunk_size = max(1, len(valid_codes) // MAX_WORKERS)
        chunks = [
            valid_codes[i:i + chunk_size]
            for i in range(0, len(valid_codes), chunk_size)
        ]

        tasks = [
            (best_indicator, chunk, start_date, effective_end,
             self._data_dir, benchmark_return)
            for chunk in chunks
        ]

        all_results = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(_worker_select_stocks, task)
                for task in tasks
            ]
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.warning(f'选股 chunk 失败: {e}')

        # 按超额收益降序
        all_results.sort(key=lambda x: x['excess_return'], reverse=True)

        top_n = min(TOP_N_STOCKS, len(all_results))
        top_stocks = [r['stock'] for r in all_results[:top_n]]

        print(f'\n  🏆 Top {top_n} 股票:')
        for i, r in enumerate(all_results[:top_n], 1):
            print(f'  #{i:2d} {r["stock"]}  超额收益: {r["excess_return"]:.2%}')

        return top_stocks, all_results

    # ==================================================================
    # Phase 3: 验证（仅 10 只股票，串行即可）
    # ==================================================================

    def verify_selection(
        self,
        top_stocks: List[str],
        best_indicator: str,
    ) -> Tuple[bool, List[Dict]]:
        """验证选股在近 3 个月是否仍然有效。"""
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=VERIFY_MONTHS * 31)
        verify_start = start_dt.strftime('%Y-%m-%d')
        verify_end = end_dt.strftime('%Y-%m-%d')

        effective_end = verify_end
        try:
            if date.fromisoformat(effective_end) >= date.today():
                effective_end = ''
        except (ValueError, TypeError):
            pass

        print(f'\n{"=" * 60}')
        print(f'[Phase 3] 验证 — 指标: {best_indicator}')
        print(f'  股票数: {len(top_stocks)}')
        print(f'  验证期: {verify_start} → {verify_end}')
        print(f'{"=" * 60}')

        benchmark_return = self._get_benchmark_return(verify_start, verify_end)
        print(f'  基准（沪深300）收益率: {benchmark_return:.2%}')

        details = []
        passed_count = 0

        for code in top_stocks:
            try:
                df = self.data_loader.load_stock(
                    code, verify_start, effective_end, min_days=MIN_TRADING_DAYS,
                )
            except Exception:
                df = None

            if df is None or len(df) < 60:
                details.append({
                    'stock': code, 'excess_return': 0,
                    'beat_benchmark': False, 'error': '数据不可用',
                })
                print(f'  ✗ {code}  数据不可用')
                continue

            try:
                excess = _worker_run_backtest(df, best_indicator, benchmark_return)
                beat = excess > 0
                if beat:
                    passed_count += 1
                details.append({
                    'stock': code,
                    'excess_return': round(float(excess), 6),
                    'beat_benchmark': beat,
                })
                status = '✓' if beat else '✗'
                print(f'  {status} {code}  超额收益: {excess:.2%}')
            except Exception as e:
                details.append({
                    'stock': code, 'excess_return': 0,
                    'beat_benchmark': False, 'error': str(e),
                })
                print(f'  ✗ {code}  错误: {e}')

        passed = passed_count >= VERIFY_PASS_THRESHOLD

        print(f'\n  结果: {passed_count}/{len(top_stocks)} 跑赢基准')
        if passed:
            print(f'  ✅ 验证通过！进入模拟盘阶段')
        else:
            print(f'  ❌ 验证失败（需 ≥{VERIFY_PASS_THRESHOLD}/{TOP_N_STOCKS}），'
                  f'需要重新扫描')

        return passed, details

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _get_benchmark_return(
        self,
        start_date: str,
        end_date: str,
    ) -> float:
        """计算基准指数的总收益率。"""
        bench_df = load_index_data(start_date, end_date)
        if bench_df is not None and len(bench_df) > 0:
            return float(
                bench_df['close'].iloc[-1] / bench_df['close'].iloc[0] - 1
            )
        return 0.0

    @staticmethod
    def _print_top_scores(scores: List[Dict]):
        """打印 Top 10 指标评分。"""
        print(f'\n{"─" * 50}')
        print(f'  🏆 最佳指标 Top 10（综合评分）')
        print(f'{"─" * 50}')
        for i, s in enumerate(scores[:10], 1):
            print(f'  #{i:2d} {s["indicator"]:12s}  '
                  f'score={s["score"]:.4f}  '
                  f'mean_ex={s["mean_excess"]:.3%}  '
                  f'win={s["win_rate"]:.0%}  '
                  f'n={s["num_valid"]}')
