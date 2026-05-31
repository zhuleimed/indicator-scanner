"""
扫描引擎 — 015_indicator_scanner

Phase 1: 全指标扫描 → 找到最稳健的普适性指标
Phase 2: 用最佳指标选 Top N 股票
Phase 3: 验证选股在近 3 个月是否仍有效

核心设计：
  - 轻量回测：不写文件、不画图、不打印日志
  - 扁平并行：所有 (indicator, stock) 对一次性提交到线程池
  - 数据预加载：300 只股票的 DataFrame 一次读入内存
"""

import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
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
from core.metrics import MetricsCalculator
from core.log_utils import get_logger
from core.hs300_utils import load_index_data
from signals.gf import GFSignal

logger = get_logger(__name__)


class ScannerEngine:
    """Phase 1-3 扫描编排器"""

    def __init__(self, data_dir: str = None):
        self.data_loader = DataLoader(data_dir)
        self._stock_cache: Dict[str, pd.DataFrame] = {}

    # ==================================================================
    # Phase 1: 全指标扫描
    # ==================================================================

    def scan_all_indicators(
        self,
        stock_codes: List[str],
        start_date: str = '',
        end_date: str = '',
    ) -> List[Dict]:
        """
        扫描全部 97 个指标，找出最稳健的普适性指标。

        Parameters
        ----------
        stock_codes : list of str
            沪深300成分股代码列表
        start_date, end_date : str
            回测日期范围（空=自动计算）

        Returns
        -------
        list of dict
            按综合评分降序排列的指标列表
            [{indicator, score, mean_excess, win_rate, std_excess, num_valid}]
        """
        indicators = list(GFSignal.INDICATORS)

        if not start_date:
            # 默认：2 年前到今天
            end_dt = date.today()
            start_dt = end_dt.replace(year=end_dt.year - SCAN_YEARS)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        print(f'\n{"=" * 60}')
        print(f'[Phase 1] 全指标扫描')
        print(f'  指标数: {len(indicators)}')
        print(f'  股票数: {len(stock_codes)}')
        print(f'  回测期: {start_date} → {end_date}')
        print(f'  并行:   {MAX_WORKERS} 线程')
        print(f'{"=" * 60}')

        t0 = time.time()

        # ---- 预加载股票数据 ----
        print('\n[1/3] 预加载股票数据…')
        self._preload_stocks(stock_codes, start_date, end_date)
        valid_codes = list(self._stock_cache.keys())
        print(f'  有效股票: {len(valid_codes)}/{len(stock_codes)}')

        if len(valid_codes) < 50:
            print('  ⚠ 有效股票过少，扫描结果可能不可靠')

        # ---- 计算基准收益 ----
        print('\n[2/3] 加载基准指数…')
        benchmark_return = self._get_benchmark_return(start_date, end_date)
        print(f'  基准（沪深300）收益率: {benchmark_return:.2%}')

        # ---- 并行扫描 ----
        print(f'\n[3/3] 并行扫描 {len(indicators)} × {len(valid_codes)} 组合…')
        total_tasks = len(indicators) * len(valid_codes)

        # 按 indicator 分组的结果收集
        indicator_results: Dict[str, List[float]] = defaultdict(list)
        lock = threading.Lock()
        completed = [0]
        last_report = [time.time()]

        def run_one(indicator: str, code: str) -> Optional[Tuple[str, float]]:
            """单个 (indicator, stock) 对的轻量回测。"""
            try:
                df = self._stock_cache[code].copy()
                excess = self._run_pair(df, indicator, benchmark_return)
                return (indicator, excess)
            except Exception as e:
                # 静默跳过（ST、数据不足等）
                return None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for indicator in indicators:
                for code in valid_codes:
                    futures.append(executor.submit(run_one, indicator, code))

            for future in as_completed(futures):
                result = future.result()
                with lock:
                    completed[0] += 1
                    if result is not None:
                        indicator, excess = result
                        indicator_results[indicator].append(excess)

                    # 每 5 秒汇报一次进度
                    now = time.time()
                    if now - last_report[0] >= 5:
                        pct = completed[0] / total_tasks * 100
                        elapsed = now - t0
                        rate = completed[0] / elapsed if elapsed > 0 else 0
                        eta = (total_tasks - completed[0]) / rate if rate > 0 else 0
                        print(f'  进度: {completed[0]:,}/{total_tasks:,} '
                              f'({pct:.1f}%)  速率: {rate:.0f}/s  '
                              f'ETA: {eta:.0f}s', end='\r')
                        last_report[0] = now

        elapsed = time.time() - t0
        print(f'\n  完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)')

        # ---- 计算综合评分 ----
        print('\n计算综合评分…')
        scores = []
        for indicator in indicators:
            excess_list = indicator_results.get(indicator, [])
            num_valid = len(excess_list)
            if num_valid < 10:
                # 有效样本太少，跳过
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
        print(f'\n{"─" * 50}')
        print(f'  🏆 最佳指标 Top 10（综合评分）')
        print(f'{"─" * 50}')
        for i, s in enumerate(scores[:10], 1):
            print(f'  #{i:2d} {s["indicator"]:12s}  '
                  f'score={s["score"]:.4f}  '
                  f'mean_ex={s["mean_excess"]:.3%}  '
                  f'win={s["win_rate"]:.0%}  '
                  f'n={s["num_valid"]}')

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
        """
        用最佳指标对全部股票回测，选出超额收益最高的 Top N 股票。

        Parameters
        ----------
        stock_codes : list of str
        best_indicator : str
            Phase 1 选出的最佳指标
        start_date, end_date : str

        Returns
        -------
        top_stocks : list of str
            Top N 股票代码
        rankings : list of dict
            完整排名
        """
        if not start_date:
            end_dt = date.today()
            start_dt = end_dt.replace(year=end_dt.year - SCAN_YEARS)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        print(f'\n{"=" * 60}')
        print(f'[Phase 2] 选股 — 指标: {best_indicator}')
        print(f'  股票数: {len(stock_codes)}')
        print(f'{"=" * 60}')

        # 预加载（如果还没加载）
        if not self._stock_cache:
            self._preload_stocks(stock_codes, start_date, end_date)

        benchmark_return = self._get_benchmark_return(start_date, end_date)

        valid_codes = list(self._stock_cache.keys())
        results = []
        lock = threading.Lock()

        print(f'  并行评估 {len(valid_codes)} 只股票…')

        def run_one(code: str) -> Optional[Dict]:
            try:
                df = self._stock_cache[code].copy()
                excess = self._run_pair(df, best_indicator, benchmark_return)
                return {'stock': code, 'excess_return': round(excess, 6)}
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_one, c): c for c in valid_codes}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    with lock:
                        results.append(result)

        # 按超额收益降序
        results.sort(key=lambda x: x['excess_return'], reverse=True)

        top_n = min(TOP_N_STOCKS, len(results))
        top_stocks = [r['stock'] for r in results[:top_n]]

        print(f'\n  🏆 Top {top_n} 股票:')
        for i, r in enumerate(results[:top_n], 1):
            print(f'  #{i:2d} {r["stock"]}  超额收益: {r["excess_return"]:.2%}')

        return top_stocks, results

    # ==================================================================
    # Phase 3: 验证
    # ==================================================================

    def verify_selection(
        self,
        top_stocks: List[str],
        best_indicator: str,
    ) -> Tuple[bool, List[Dict]]:
        """
        验证选股在近 3 个月是否仍然有效。

        Parameters
        ----------
        top_stocks : list of str
            Phase 2 选出的股票
        best_indicator : str

        Returns
        -------
        passed : bool
            是否通过验证
        details : list of dict
            每只股票的验证详情
        """
        end_dt = date.today()
        # 近 3 个月：直接减去 ~93 天
        from datetime import timedelta
        start_dt = end_dt - timedelta(days=VERIFY_MONTHS * 31)
        verify_start = start_dt.strftime('%Y-%m-%d')
        verify_end = end_dt.strftime('%Y-%m-%d')

        print(f'\n{"=" * 60}')
        print(f'[Phase 3] 验证 — 指标: {best_indicator}')
        print(f'  股票数: {len(top_stocks)}')
        print(f'  验证期: {verify_start} → {verify_end}')
        print(f'{"=" * 60}')

        # 加载验证期数据
        self._stock_cache.clear()  # 重新加载较短的时间段
        self._preload_stocks(top_stocks, verify_start, verify_end)

        benchmark_return = self._get_benchmark_return(verify_start, verify_end)
        print(f'  基准（沪深300）收益率: {benchmark_return:.2%}')

        details = []
        passed_count = 0

        for code in top_stocks:
            if code not in self._stock_cache:
                details.append({
                    'stock': code,
                    'excess_return': 0,
                    'beat_benchmark': False,
                    'error': '数据不可用',
                })
                continue

            try:
                df = self._stock_cache[code].copy()
                excess = self._run_pair(df, best_indicator, benchmark_return)
                beat = excess > 0
                if beat:
                    passed_count += 1
                details.append({
                    'stock': code,
                    'excess_return': round(excess, 6),
                    'beat_benchmark': beat,
                })
                status = '✓' if beat else '✗'
                print(f'  {status} {code}  超额收益: {excess:.2%}')
            except Exception as e:
                details.append({
                    'stock': code,
                    'excess_return': 0,
                    'beat_benchmark': False,
                    'error': str(e),
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

    def _preload_stocks(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str = '',
    ):
        """预加载所有股票的 DataFrame 到缓存。"""
        self._stock_cache.clear()

        # 如果 end_date 是今天或未来日期，使用空字符串让 loader 自动取到最新数据
        effective_end = end_date
        if effective_end:
            from datetime import date as _date
            try:
                end_dt = _date.fromisoformat(effective_end)
                if end_dt >= _date.today():
                    effective_end = ''  # 让 loader 使用数据中的最新日期
            except (ValueError, TypeError):
                pass

        for code in stock_codes:
            try:
                df = self.data_loader.load_stock(
                    code, start_date, effective_end, min_days=MIN_TRADING_DAYS,
                )
                if df is not None and len(df) >= 60:
                    self._stock_cache[code] = df
            except Exception:
                pass

    def _get_benchmark_return(
        self,
        start_date: str,
        end_date: str,
    ) -> float:
        """计算基准指数的总收益率。"""
        bench_df = load_index_data(start_date, end_date)
        if bench_df is not None and len(bench_df) > 0:
            bench_ret = (
                bench_df['close'].iloc[-1] / bench_df['close'].iloc[0] - 1
            )
            return bench_ret
        return 0.0

    def _run_pair(
        self,
        stock_df: pd.DataFrame,
        indicator: str,
        benchmark_return: float,
    ) -> float:
        """
        对单个 (indicator, stock) 对执行轻量回测，
        返回超额收益率。

        这是整个扫描系统最底层的计算单元，
        被调用了 ~29,000 次，必须高效。
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

        # 信号修正：卖出信号覆盖 stop_signal
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

        # ---- 6. 超额收益 ----
        excess_return = strategy_return - benchmark_return
        return excess_return
