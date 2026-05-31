"""
消息推送模块 — 015_indicator_scanner

职责：
  1. 每日模拟盘日报 → WxPusher 微信推送
  2. 扫描完成通知
  3. 异常告警
"""

from datetime import date
from typing import Any, Dict, List, Optional

from wxpusher import WxPusher

from config.scanner_config import (
    WXPUSHER_TOKEN,
    WXPUSHER_UIDS,
    WXPUSHER_TOPIC_IDS,
)


def _send(message: str):
    """底层发送，失败不抛异常。"""
    try:
        WxPusher.send_message(
            message,
            uids=WXPUSHER_UIDS,
            topic_ids=WXPUSHER_TOPIC_IDS,
            token=WXPUSHER_TOKEN,
        )
    except Exception as e:
        print(f'[WxPusher] 推送失败: {e}')


def push_daily_report(summary: Dict[str, Any]):
    """
    推送每日模拟盘日报。

    Parameters
    ----------
    summary : dict
        Simulator.run_daily() 返回的日报摘要
    """
    today = summary.get('date', date.today().isoformat())

    lines = [
        f'📊 指标扫描 · 模拟盘日报',
        f'日期: {today}',
        f'指标: {summary.get("indicator", "N/A")}',
    ]

    if summary.get('dry_run'):
        lines.append('⚠ DRY RUN 模式（未修改状态）')

    # ---- 当日操作 ----
    trades = summary.get('trades_today', [])
    if trades:
        lines.append('')
        lines.append('── 当日操作 ──')
        for t in trades:
            if t['action'] == 'buy':
                lines.append(
                    f'🟢 买入 {t["stock"]}: '
                    f'{t["shares"]}股 @ {t["price"]:.2f}  '
                    f'成本={t["cost"]:.2f}'
                )
            else:
                lines.append(
                    f'🔴 卖出 {t["stock"]}: '
                    f'{t["shares"]}股 @ {t["price"]:.2f}  '
                    f'盈亏={t.get("pnl", 0):+.2f}  '
                    f'({t.get("reason", "")})'
                )
    else:
        lines.append('')
        lines.append('── 当日操作 ──')
        lines.append('  无操作')

    # ---- 持仓摘要 ----
    positions = summary.get('positions', [])
    lines.append('')
    lines.append('── 持仓摘要 ──')
    if positions:
        for p in positions:
            sign = '+' if p['pnl_pct'] >= 0 else ''
            lines.append(
                f'  {p["stock"]}: {p["shares"]}股  '
                f'成本={p["avg_cost"]:.2f}  '
                f'现价={p["last_close"]:.2f}  '
                f'({sign}{p["pnl_pct"]:.2%})'
            )
    else:
        lines.append('  空仓')

    # ---- 账户摘要 ----
    lines.append('')
    lines.append('── 账户摘要 ──')
    lines.append(f'总资产: {summary["portfolio_value"]:,.2f}')
    lines.append(f'现金:   {summary["cash"]:,.2f}')
    cum_ret = summary['cumulative_return']
    sign = '+' if cum_ret >= 0 else ''
    lines.append(f'累计收益: {sign}{cum_ret:.2%}')

    # ---- 明日信号预览 ----
    pending = summary.get('pending_orders', {})
    if pending:
        # 筛选非 hold 的
        active_signals = {
            s: a for s, a in pending.items() if a != 'hold'
        }
        if active_signals:
            lines.append('')
            lines.append('── 明日信号 ──')
            for stock, action in active_signals.items():
                emoji = '🟢' if action == 'buy' else '🔴'
                lines.append(f'  {emoji} {stock}: {action}')

    _send('\n'.join(lines))


def push_scan_complete(
    best_indicator: str,
    top_stocks: List[str],
    scan_scores: Optional[List[Dict]] = None,
):
    """
    推送扫描完成通知。

    Parameters
    ----------
    best_indicator : str
        选出的最佳指标
    top_stocks : list of str
        Top 10 股票
    scan_scores : list of dict, optional
        完整评分排行（仅展示 Top 5）
    """
    lines = [
        '🔍 指标扫描完成',
        f'日期: {date.today().isoformat()}',
        '',
        f'🏆 最佳指标: {best_indicator}',
        '',
        f'📈 Top 10 股票:',
    ]
    for i, s in enumerate(top_stocks, 1):
        lines.append(f'  #{i} {s}')

    if scan_scores:
        lines.append('')
        lines.append('📊 指标评分 Top 5:')
        for i, s in enumerate(scan_scores[:5], 1):
            lines.append(
                f'  #{i} {s["indicator"]}: '
                f'score={s["score"]:.4f}  '
                f'win={s["win_rate"]:.0%}'
            )

    _send('\n'.join(lines))


def push_error(error_msg: str, phase: str = ''):
    """
    推送异常告警。

    Parameters
    ----------
    error_msg : str
        错误信息
    phase : str
        出错阶段
    """
    lines = [
        '⚠ 指标扫描 · 异常告警',
        f'日期: {date.today().isoformat()}',
    ]
    if phase:
        lines.append(f'阶段: {phase}')
    lines.append('')
    lines.append(f'错误: {error_msg}')

    _send('\n'.join(lines))
