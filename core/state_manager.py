"""
状态管理模块 — 015_indicator_scanner

职责：
  1. JSON 状态的持久化（原子写入）
  2. 状态机流转控制
  3. 持仓 / 交易记录 / 扫描结果的 CRUD

状态文件 schema 见 CLAUDE.md 或本文件 _default_state() 方法。
"""

import json
import os
import tempfile
from datetime import datetime, date
from typing import Any, Dict, List, Optional


class StateManager:
    """回测扫描器状态管理器"""

    def __init__(self, state_file_path: str):
        self._path = state_file_path
        self._data: Dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """从文件加载状态。文件不存在时创建默认状态。"""
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                # 向后兼容：补齐缺失字段
                default = self._default_state()
                for key, val in default.items():
                    if key not in self._data:
                        self._data[key] = val
            except (json.JSONDecodeError, IOError) as e:
                print(f'[state] 状态文件损坏，使用默认状态: {e}')
                self._data = self._default_state()
        else:
            self._data = self._default_state()
        return self._data

    def save(self):
        """原子写入状态文件（先写临时文件再重命名）。"""
        self._data['last_update_date'] = date.today().isoformat()

        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix='.json',
            prefix='state_',
            dir=os.path.dirname(self._path),
        )
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2,
                          default=str)
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # 状态数据访问
    # ------------------------------------------------------------------

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def current_phase(self) -> str:
        return self._data.get('current_phase', 'idle')

    @current_phase.setter
    def current_phase(self, phase: str):
        valid = {'idle', 'scanning', 'selecting', 'verifying', 'running'}
        if phase not in valid:
            raise ValueError(f'无效 phase: {phase}，可选: {valid}')
        self._data['current_phase'] = phase

    @property
    def best_indicator(self) -> Optional[str]:
        return self._data.get('best_indicator')

    @property
    def top_10_stocks(self) -> List[str]:
        return self._data.get('top_10_stocks', [])

    @property
    def portfolio(self) -> Dict[str, Any]:
        return self._data.get('portfolio', {})

    # ------------------------------------------------------------------
    # Phase 1: 扫描结果
    # ------------------------------------------------------------------

    def set_scan_results(self,
                         scan_results: List[Dict],
                         best_indicator: str):
        """
        保存 Phase 1 扫描结果。

        Parameters
        ----------
        scan_results : list of dict
            每个指标的综合评分 [{indicator, score, mean_excess, win_rate, ...}]
        best_indicator : str
            评分最高的指标名称
        """
        self._data['scan_results'] = scan_results
        self._data['best_indicator'] = best_indicator
        self._data['scan_date'] = datetime.now().isoformat()
        self._data['last_scan_date'] = date.today().isoformat()
        self._data['current_phase'] = 'scanning'
        self.save()

    # ------------------------------------------------------------------
    # Phase 2: 选股结果
    # ------------------------------------------------------------------

    def set_top_stocks(self,
                       stocks: List[str],
                       rankings: Optional[List[Dict]] = None):
        """
        保存 Phase 2 选股结果。

        Parameters
        ----------
        stocks : list of str
            Top N 股票代码列表（按超额收益降序）
        rankings : list of dict, optional
            完整排名 [{stock, excess_return, total_return, benchmark_return}]
        """
        self._data['top_10_stocks'] = stocks
        if rankings is not None:
            self._data['stock_rankings'] = rankings
        self._data['current_phase'] = 'selecting'
        self.save()

    # ------------------------------------------------------------------
    # Phase 3: 验证结果
    # ------------------------------------------------------------------

    def set_verify_results(self, passed: bool,
                           details: List[Dict]):
        """
        保存 Phase 3 验证结果。

        Parameters
        ----------
        passed : bool
            是否通过验证（≥6/10 跑赢基准）
        details : list of dict
            每只股票的验证详情
        """
        self._data['verify_results'] = {
            'date': date.today().isoformat(),
            'passed': passed,
            'details': details,
            'num_passed': sum(1 for d in details if d.get('beat_benchmark', False)),
            'num_total': len(details),
        }
        if passed:
            self._data['current_phase'] = 'running'
            # 初始化模拟盘投资组合
            if 'portfolio' not in self._data or not self._data['portfolio'].get('initial_capital'):
                self._data['portfolio'] = {
                    'cash': self._data.get('portfolio', {}).get(
                        'initial_capital', 1_000_000.0
                    ),
                    'initial_capital': self._data.get('portfolio', {}).get(
                        'initial_capital', 1_000_000.0
                    ),
                    'positions': {},
                    'pending_orders': {},
                }
        else:
            self._data['current_phase'] = 'idle'
        self._data['verify_passed'] = passed
        self.save()

    # ------------------------------------------------------------------
    # Phase 4: 模拟盘持仓更新
    # ------------------------------------------------------------------

    def update_portfolio(self,
                         cash: float,
                         positions: Dict[str, Any],
                         pending_orders: Dict[str, str]):
        """
        更新模拟盘持仓状态。

        Parameters
        ----------
        cash : float
            当前现金
        positions : dict
            当前持仓 {stock_code: {shares, avg_cost, total_cost}}
        pending_orders : dict
            明日待执行订单 {stock_code: 'buy'|'sell'|'hold'}
        """
        pf = self._data.setdefault('portfolio', {})
        pf['cash'] = round(cash, 2)
        pf['positions'] = positions
        pf['pending_orders'] = pending_orders
        # 保留 initial_capital
        if 'initial_capital' not in pf:
            pf['initial_capital'] = 1_000_000.0

    def add_trade(self, entry: Dict):
        """
        追加交易记录。

        Parameters
        ----------
        entry : dict
            {date, stock, action, price, shares, pnl, reason}
        """
        log = self._data.setdefault('trade_log', [])
        log.append(entry)
        # 仅保留最近 180 天
        if len(log) > 500:
            self._data['trade_log'] = log[-500:]

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def needs_rescan(self) -> bool:
        """
        判断是否需要重新扫描。

        Returns
        -------
        bool
            - 从未扫描过 → True
            - 距上次扫描超过 RESCAN_DAYS 天 → True
            - 当前 phase 为 idle → True
            - 否则 → False
        """
        from config.scanner_config import RESCAN_DAYS

        if self.current_phase == 'idle':
            return True

        last_scan = self._data.get('last_scan_date')
        if last_scan is None:
            return True

        try:
            last = datetime.strptime(str(last_scan)[:10], '%Y-%m-%d').date()
            return (date.today() - last).days >= RESCAN_DAYS
        except (ValueError, TypeError):
            return True

    def is_first_run(self) -> bool:
        """是否是首次运行（从未有过扫描结果）。"""
        return self._data.get('scan_date') is None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            'version': 1,
            'current_phase': 'idle',
            'best_indicator': None,
            'scan_date': None,
            'last_scan_date': None,
            'scan_results': [],
            'top_10_stocks': [],
            'stock_rankings': [],
            'verify_passed': False,
            'verify_results': None,
            'portfolio': {
                'cash': 1_000_000.0,
                'initial_capital': 1_000_000.0,
                'positions': {},
                'pending_orders': {},
            },
            'trade_log': [],
            'last_update_date': None,
        }
