#!/usr/bin/env python3
"""
run_scanner.py — 指标扫描器主入口

用法:
  python run_scanner.py                  # --phase auto（默认，自动判断）
  python run_scanner.py --force          # 强制重新扫描
  python run_scanner.py --phase 4        # 仅执行模拟盘
  python run_scanner.py --dry-run        # 模拟盘 dry-run（不修改状态）
  python run_scanner.py --test-mode      # 小规模测试（3指标×10股票）

Cron 配置示例:
  # 每日模拟盘（交易日 21:00，周一至周五）
  0 21 * * 1-5 cd /path/to/015_indicator_scanner && \\
    /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py >> logs/daily_\$(date +\%Y\%m\%d).log 2>&1

  # 每季度扫描（3/6/9/12月1日凌晨2:00）
  0 2 1 3,6,9,12 * cd /path/to/015_indicator_scanner && \\
    /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py --force >> logs/quarterly_\$(date +\%Y\%m\%d).log 2>&1
"""

import argparse
import os
import sys
from datetime import date, datetime

# 确保项目根目录在 sys.path 中
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config.scanner_config import (
    ensure_dirs, OUTPUT_DIR, STATE_FILE,
    TOP_N_STOCKS,
)
from core.state_manager import StateManager
from core.scanner import ScannerEngine
from core.simulator import Simulator
from core.notification import (
    push_daily_report, push_scan_complete, push_error,
)
from core.hs300_utils import fetch_hs300_constituents, is_trading_day
from core.log_utils import get_logger

logger = get_logger(__name__)


def run_scan_pipeline(
    state: StateManager,
    scanner: ScannerEngine,
    test_mode: bool = False,
):
    """
    执行 Phase 1 → 2 → 3 完整扫描流水线。

    成功 → state 变为 'running'
    失败（Phase 3 验证不通过）→ state 变为 'idle'
    """
    try:
        # ---- Phase 1: 扫描 ----
        state.current_phase = 'scanning'
        state.save()

        stock_codes = fetch_hs300_constituents()
        if not stock_codes:
            push_error('无法获取沪深300成分股列表', 'Phase 1')
            return

        if test_mode:
            # 小规模测试：仅用前 10 只股票 + 3 个指标
            stock_codes = stock_codes[:10]
            # 临时限制指标数
            from signals.gf import GFSignal
            orig_indicators = GFSignal.INDICATORS
            # 只用 KDJ, MACD, RSI 做快速测试
            GFSignal.INDICATORS = ['KDJ', 'MACD', 'RSI', 'CCI', 'WR']

        scan_results = scanner.scan_all_indicators(stock_codes)

        if test_mode:
            GFSignal.INDICATORS = orig_indicators

        if not scan_results:
            push_error('扫描失败：无有效结果', 'Phase 1')
            return

        best_indicator = scan_results[0]['indicator']
        state.set_scan_results(scan_results, best_indicator)
        logger.info(f'Phase 1 完成: 最佳指标 = {best_indicator}')

        # ---- Phase 2: 选股 ----
        state.current_phase = 'selecting'
        state.save()

        top_stocks, rankings = scanner.select_top_stocks(
            stock_codes, best_indicator,
        )
        state.set_top_stocks(top_stocks, rankings)
        logger.info(f'Phase 2 完成: Top {len(top_stocks)} 股票')

        # ---- Phase 3: 验证 ----
        state.current_phase = 'verifying'
        state.save()

        passed, details = scanner.verify_selection(
            top_stocks, best_indicator,
        )
        state.set_verify_results(passed, details)

        if passed:
            logger.info('Phase 3 通过！进入模拟盘阶段')
            push_scan_complete(best_indicator, top_stocks, scan_results)
        else:
            logger.warning('Phase 3 未通过，状态重置为 idle')
            push_error(
                f'验证未通过: {sum(1 for d in details if d.get("beat_benchmark"))}/'
                f'{len(details)} 跑赢基准',
                'Phase 3',
            )

    except Exception as e:
        logger.error(f'扫描流水线异常: {e}', exc_info=True)
        push_error(str(e), '扫描流水线')
        state.current_phase = 'idle'
        state.save()


def run_daily_simulation(
    state: StateManager,
    simulator: Simulator,
    dry_run: bool = False,
):
    """执行 Phase 4 每日模拟盘。"""
    try:
        best_indicator = state.best_indicator
        top_stocks = state.top_10_stocks

        if not best_indicator or not top_stocks:
            push_error('状态数据不完整：缺少最佳指标或股票列表', 'Phase 4')
            return

        summary = simulator.run_daily(
            top_stocks=top_stocks,
            best_indicator=best_indicator,
            state_manager=state,
            dry_run=dry_run,
        )

        if summary is None:
            # 非交易日
            return

        if not dry_run:
            push_daily_report(summary)

    except Exception as e:
        logger.error(f'模拟盘异常: {e}', exc_info=True)
        push_error(str(e), 'Phase 4')


def main():
    parser = argparse.ArgumentParser(
        description='015_indicator_scanner — 指标扫描与模拟盘系统',
    )
    parser.add_argument(
        '--phase', type=str, default='auto',
        choices=['auto', '1', '2', '3', '4'],
        help='指定执行阶段。auto=自动判断（默认）',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制重新扫描（忽略当前状态）',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='模拟盘 dry-run 模式（不修改状态文件）',
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help='小规模测试模式（仅用少量股票和指标）',
    )
    args = parser.parse_args()

    # ---- 初始化 ----
    ensure_dirs()
    state = StateManager(STATE_FILE)
    scanner = ScannerEngine()
    simulator = Simulator()

    # ---- 确定执行阶段 ----
    if args.force:
        phase = 'scan'
    elif args.phase == 'auto':
        if state.needs_rescan():
            phase = 'scan'
        elif state.current_phase == 'running':
            if is_trading_day():
                phase = '4'
            else:
                logger.info(f'非交易日，跳过模拟盘')
                return
        else:
            # 状态异常，回退到扫描
            logger.warning(f'状态异常 (phase={state.current_phase})，执行扫描')
            phase = 'scan'
    elif args.phase == '1':
        phase = 'scan_one'
    elif args.phase == '2':
        phase = 'scan_two'
    elif args.phase == '3':
        phase = 'scan_three'
    else:
        phase = args.phase

    # ---- 执行 ----
    if phase == 'scan':
        run_scan_pipeline(state, scanner, test_mode=args.test_mode)
    elif phase == 'scan_one':
        state.current_phase = 'scanning'
        state.save()
        stock_codes = fetch_hs300_constituents()
        if args.test_mode:
            stock_codes = stock_codes[:10]
        scan_results = scanner.scan_all_indicators(stock_codes)
        best = scan_results[0]['indicator']
        state.set_scan_results(scan_results, best)
        print(f'Phase 1 完成: {best}')
    elif phase == 'scan_two':
        stock_codes = fetch_hs300_constituents()
        if args.test_mode:
            stock_codes = stock_codes[:10]
        best_indicator = state.best_indicator
        if not best_indicator:
            print('错误：请先执行 Phase 1')
            sys.exit(1)
        top_stocks, rankings = scanner.select_top_stocks(
            stock_codes, best_indicator,
        )
        state.set_top_stocks(top_stocks, rankings)
        print(f'Phase 2 完成: {top_stocks}')
    elif phase == 'scan_three':
        top_stocks = state.top_10_stocks
        best_indicator = state.best_indicator
        if not top_stocks or not best_indicator:
            print('错误：请先执行 Phase 1 和 Phase 2')
            sys.exit(1)
        passed, details = scanner.verify_selection(top_stocks, best_indicator)
        state.set_verify_results(passed, details)
        print(f'Phase 3 完成: {"通过" if passed else "未通过"}')
    elif phase == '4':
        run_daily_simulation(state, simulator, dry_run=args.dry_run)

    logger.info('执行完毕')


if __name__ == '__main__':
    main()
