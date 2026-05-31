"""
回测引擎主模块 (BacktestEngine)

职责：
  作为回测流程的总编排器，将以下模块串联为完整的流水线：
    DataLoader → SignalEngine → RiskManager → EquityCurveCalculator → MetricsCalculator → Reporter

  优化功能：
    - 并行处理：多股票回测时使用进程池加速
    - 进度显示：tqdm 进度条
    - 日志系统：logging 替代 print
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .data_loader import DataLoader
from .signal_engine import SignalEngine, BaseSignal
from .risk_manager import RiskManager
from .equity_curve import BacktestParams, EquityCurveCalculator, TradeRecord
from .metrics import BacktestMetrics, MetricsCalculator
from .reporter import Reporter
from .log_utils import get_logger, console_out

logger = get_logger(__name__)


def _worker_compute_equity(args: tuple):
    """
    模块级 worker 函数：在子进程中计算单只股票的资金曲线。

    ProcessPoolExecutor 要求目标函数在模块级别定义（可 pickle）。
    所有参数通过 args tuple 传入，避免闭包捕获 self。
    """
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    np.seterr(all='ignore')

    (stock_df, initial_money, slippage, commission_rate,
     tax_rate, position_pct) = args
    params = BacktestParams(
        initial_money=initial_money,
        slippage=slippage,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        position_pct=position_pct,
    )
    calc = EquityCurveCalculator(params)
    return calc.compute_single(stock_df)


@dataclass
class BacktestConfig:
    """回测完整配置"""
    # 股票与时间
    stock_codes: List[str] = field(default_factory=list)
    start_date: str = ''
    end_date: str = ''
    benchmark_code: str = 'sh.000300'  # 沪深300

    # 资金与交易参数
    initial_money_per_stock: float = 10_000.0
    slippage: float = 0.003
    commission_rate: float = 0.0005
    tax_rate: float = 0.001
    position_pct: float = 0.95
    risk_free_rate: float = 0.027

    # 风控参数
    stop_loss_pct: float = 0.05
    stop_profit_pct: float = 0.20
    drawdown_pct: float = 0.03
    # 路径
    data_dir: Optional[str] = None
    output_dir: str = ''

    # 报告
    tag: str = ''

    # 并行
    max_workers: int = 4               # 并行回测线程数

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'output'
            )
        if self.start_date and isinstance(self.start_date, str):
            self.start_date = self.start_date


class BacktestEngine:
    """回测引擎——流水线总编排器"""

    def __init__(self, config: BacktestConfig):
        self.config = config

        self.data_loader = DataLoader(config.data_dir)
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        trade_params = BacktestParams(
            initial_money=config.initial_money_per_stock,
            slippage=config.slippage,
            commission_rate=config.commission_rate,
            tax_rate=config.tax_rate,
            position_pct=config.position_pct,
        )
        self.equity_calculator = EquityCurveCalculator(trade_params)
        self.metrics_calculator = MetricsCalculator(config.risk_free_rate)
        self.reporter = Reporter(config.output_dir)

        self._stock_datas: Dict[str, pd.DataFrame] = {}
        self._stock_results: List[Dict] = []
        self._benchmark_data: Optional[pd.DataFrame] = None
        self._account_data: Optional[pd.DataFrame] = None
        self._metrics: Optional[BacktestMetrics] = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def register_signal(self, strategy: BaseSignal,
                         weight: float = 1.0) -> 'BacktestEngine':
        self.signal_engine.register(strategy, weight)
        return self

    @property
    def signal_name(self) -> str:
        if self.signal_engine.registered:
            return self.signal_engine.registered[0].name
        return 'strategy'

    def run(self) -> BacktestMetrics:
        """
        执行完整回测流水线：
          数据加载 → 信号生成 → 风控 → 资金曲线 → 绩效指标 → 报告
        """
        cfg = self.config
        tag = cfg.tag or self.signal_name

        output_subdir = cfg.output_dir
        os.makedirs(output_subdir, exist_ok=True)
        self.reporter.output_dir = output_subdir

        console_out(f'')
        console_out(f'{"=" * 50}')
        logger.info(f'开始回测: {tag}')
        logger.info(f'  股票池: {len(cfg.stock_codes)} 只')
        logger.info(f'  时间:   {cfg.start_date} → {cfg.end_date or "最近"}')
        logger.info(f'  策略:   {self.signal_name}')
        logger.info(f'  输出:   {output_subdir}')
        console_out(f'{"=" * 50}')

        # ---- Step 1: 数据加载 ----
        logger.info('[1/5] 加载数据…')
        self._load_stock_data()
        self._load_benchmark_data()
        logger.info(f'  ✓ 加载 {len(self._stock_datas)} 只股票, 1 个基准指数')

        # ---- Step 2: 信号生成 ----
        logger.info('[2/5] 生成交易信号…')
        self._generate_signals()
        logger.info(f'  ✓ 信号策略: {self.signal_name}')

        # ---- Step 3: 风控 ----
        logger.info('[3/5] 施加风控规则…')
        self._apply_risk_controls()

        # ---- Step 4: 资金曲线（并行）----
        logger.info('[4/5] 计算资金曲线…')
        self._calculate_equity_curves()

        # ---- Step 5: 绩效与报告 ----
        logger.info('[5/5] 计算绩效 & 生成报告…')
        self._compute_metrics()
        self._generate_reports(tag)

        self.reporter.print_summary(self._metrics)
        logger.info(f'回测完成 ✓')
        return self._metrics

    # ------------------------------------------------------------------
    # 内部步骤实现
    # ------------------------------------------------------------------

    def _load_stock_data(self):
        cfg = self.config
        self._stock_datas = self.data_loader.load_stock_batch(
            cfg.stock_codes, cfg.start_date, cfg.end_date, min_days=240,
            on_error='skip',
        )

    def _load_benchmark_data(self):
        cfg = self.config
        self._benchmark_data = self.data_loader.load_benchmark(
            cfg.benchmark_code, cfg.start_date, cfg.end_date,
        )

    def _generate_signals(self):
        for code, df in self._stock_datas.items():
            self._stock_datas[code] = self.signal_engine.generate(df)

        # 消除 look-ahead bias：信号基于前一日收盘数据，在当日开盘执行
        # 向量化计算时所有行的数据（含当日 close）都在内存中，但交易仅用 open，
        # 必须将信号整体后移一天，让 pos[i] 反映 i-1 日的信号，而非 i 日。
        for code in self._stock_datas:
            df = self._stock_datas[code]
            sig_col = f'{self.signal_name}_signal'
            if sig_col in df.columns:
                df[sig_col] = df[sig_col].shift(1).fillna(0)
            df['pos'] = df['pos'].shift(1).ffill().fillna(0)

    def _apply_risk_controls(self):
        cfg = self.config
        codes = list(self._stock_datas.keys())

        for idx, code in enumerate(codes):
            df = self._stock_datas[code]

            # 涨跌停限制
            df = self.risk_manager.apply_limit_up_down(df)

            # 止盈止损
            df = self.risk_manager.apply_stop_strategy(
                df, self.signal_name,
                stop_loss_pct=cfg.stop_loss_pct,
                stop_profit_pct=cfg.stop_profit_pct,
                drawdown_pct=cfg.drawdown_pct,
            )

            # 信号修正
            for i in range(1, len(df)):
                sig_col = f'{self.signal_name}_signal'
                if df.at[i, sig_col] == -1 and df.at[i, 'stop_signal'] in [-1, -2, -3]:
                    df.at[i, 'pos'] = 0
            df['pos'] = df['pos'].ffill().fillna(0)
            self._stock_datas[code] = df

    def _calculate_equity_curves(self):
        cfg = self.config
        codes = list(self._stock_datas.keys())

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        self._stock_results = []

        if cfg.max_workers > 1 and len(codes) > 1:
            # ---- 多进程并行模式 ----
            if not has_tqdm:
                logger.info(f'  并行计算 {len(codes)} 只股票 '
                            f'({cfg.max_workers} 进程)')

            # 构建任务：将 DataFrame 和参数打包为 tuple
            tasks = [
                (self._stock_datas[code],
                 cfg.initial_money_per_stock,
                 cfg.slippage, cfg.commission_rate,
                 cfg.tax_rate, cfg.position_pct)
                for code in codes
            ]

            with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
                future_map = {
                    executor.submit(_worker_compute_equity, task): code
                    for code, task in zip(codes, tasks)
                }
                for future in as_completed(future_map):
                    result = future.result()
                    self._stock_results.append(result)

            if has_tqdm:
                # tqdm 在进程池完成后一次性标记
                pass
        else:
            # ---- 串行模式 ----
            iterator = codes
            if has_tqdm:
                iterator = tqdm(codes, desc='  资金曲线', unit='股')
            for code in iterator:
                df = self._stock_datas[code]
                result = self.equity_calculator.compute_single(df)
                self._stock_results.append(result)

        self._account_data = self.equity_calculator.compute_account(
            self._stock_results
        )

    def _compute_metrics(self):
        cfg = self.config
        all_trade_records = [r['trade_records'] for r in self._stock_results]

        benchmark_ret = None
        if self._benchmark_data is not None:
            bench = self._benchmark_data['benchmark_cumulative_returns']
            benchmark_ret = bench.iloc[-1] - 1 if len(bench) > 0 else None

        self._metrics = self.metrics_calculator.compute(
            self._account_data,
            initial_money=cfg.initial_money_per_stock * len(cfg.stock_codes),
            trade_records=all_trade_records,
            benchmark_return=benchmark_ret,
        )

    def _generate_reports(self, tag: str = '', ts: str = ''):
        cfg = self.config
        prefix = tag or 'backtest'

        all_records = [r['trade_records'] for r in self._stock_results]
        active_codes = [code for code in cfg.stock_codes if code in self._stock_datas]

        self.reporter.save_trade_records(
            all_records, active_codes,
            filename=f'{prefix}_trade_records.csv',
        )

        self.reporter.save_account_curve(
            self._account_data, self._benchmark_data,
            filename=f'{prefix}_account_curve.csv',
        )

        end_date_str = str(self._account_data['date'].iloc[-1].date())
        self.reporter.save_metrics(
            self._metrics, cfg.stock_codes, cfg.start_date, end_date_str,
            filename=f'{prefix}_metrics.csv',
        )

        codes_str = '-'.join(active_codes[:5])
        if len(active_codes) > 5:
            codes_str += f'-etc{len(active_codes)-5}'
        title = f'{tag} | {codes_str} | {cfg.start_date}→{end_date_str}'
        self.reporter.plot_returns(
            self._account_data, self._benchmark_data,
            title=title,
            filename=f'{prefix}_returns.png',
        )
