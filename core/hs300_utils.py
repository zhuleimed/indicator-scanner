"""
沪深300工具模块 — 015_indicator_scanner

职责：
  1. 从 baostock 获取沪深300成分股列表
  2. 判断当前是否为交易日
  3. 加载指数日线数据
"""

from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd

from config.scanner_config import (
    DATA_DIR, HS300_INDEX_FILE, REFERENCE_STOCK,
)


def fetch_hs300_constituents() -> List[str]:
    """
    从 baostock API 获取沪深300成分股代码列表。

    Returns
    -------
    list of str
        6位数字股票代码（已去除 'sh.' / 'sz.' 前缀）

    Notes
    -----
    成分股列表会被缓存到 output/hs300_stocks.csv，
    后续调用如果缓存未过期（< 7 天）则直接使用缓存。
    """
    import os

    cache_file = os.path.join(
        os.path.dirname(DATA_DIR), '015_indicator_scanner', 'output',
    )
    # 缓存到 output 目录下
    from config.scanner_config import OUTPUT_DIR
    cache_path = os.path.join(OUTPUT_DIR, 'hs300_stocks.csv')

    # 检查缓存是否有效（7天内）
    if os.path.exists(cache_path):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - cache_mtime).days < 7:
            df = pd.read_csv(cache_path, dtype=str)
            codes = df['code'].tolist() if 'code' in df.columns else []
            if codes:
                return codes

    import baostock as bs

    lg = bs.login()
    if lg.error_code != '0':
        print(f'[baostock] 登录失败: {lg.error_msg}')
        # 尝试使用过期缓存
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, dtype=str)
            codes = df['code'].tolist() if 'code' in df.columns else []
            if codes:
                print(f'[baostock] 使用过期缓存: {len(codes)} 只成分股')
                return codes
        return []

    rs = bs.query_hs300_stocks()
    if rs.error_code != '0':
        print(f'[baostock] 获取沪深300成分股失败: {rs.error_msg}')
        bs.logout()
        return []

    stocks = []
    while rs.next():
        row = rs.get_row_data()
        stocks.append(row)

    bs.logout()

    if not stocks:
        return []

    # rs.fields = ['updateDate', 'code', 'code_name']
    # code 格式为 'sh.600008' 或 'sz.000001'
    result = []
    for row in stocks:
        raw_code = row[1]  # 'sh.600008'
        # 去掉前三位 'sh.' 或 'sz.'
        code = raw_code.replace('sh.', '').replace('sz.', '')
        result.append(code)

    # 缓存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({'code': result}).to_csv(cache_path, index=False)

    print(f'[hs300] 获取到 {len(result)} 只沪深300成分股')
    return result


def is_trading_day() -> bool:
    """
    判断今天是否为交易日。

    通过 baostock API 的 query_trade_dates 查询，
    不依赖本地 CSV 文件，确保数据源可靠性。

    Returns
    -------
    bool
    """
    import baostock as bs

    today_str = date.today().strftime('%Y-%m-%d')

    try:
        lg = bs.login()
        if lg.error_code != '0':
            print(f'[交易日判断] baostock 登录失败，回退到工作日判断')
            return date.today().weekday() < 5

        rs = bs.query_trade_dates(start_date=today_str, end_date=today_str)
        is_trade = False
        if rs.error_code == '0':
            while rs.next():
                row = rs.get_row_data()
                # row: [calendar_date, is_trading_day]
                # is_trading_day: '1' = 交易日, '0' = 非交易日
                if len(row) >= 2 and row[0] == today_str and row[1] == '1':
                    is_trade = True

        bs.logout()
        return is_trade

    except Exception as e:
        print(f'[交易日判断] baostock 查询异常: {e}，回退到工作日判断')
        return date.today().weekday() < 5


def fetch_stock_from_baostock(
    stock_code: str,
    start_date: str,
    end_date: str,
    timeout_seconds: int = 45,
) -> Optional[pd.DataFrame]:
    """
    从 baostock API 获取单只股票的日线数据。

    用于 Phase 4 模拟盘，确保获取最新交易日数据，
    避免依赖本地 CSV 文件可能的数据缺失问题。

    Parameters
    ----------
    stock_code : str
        6位数字股票代码
    start_date : str
        开始日期 'YYYY-MM-DD'
    end_date : str
        结束日期 'YYYY-MM-DD'
    timeout_seconds : int
        单次 API 调用超时秒数（默认 45s），防止网络挂起导致脚本永久卡死

    Returns
    -------
    pd.DataFrame or None
        列与本地 CSV 格式一致：
        date, open, high, low, close, volume, amount,
        turn, peTTM, pbMRQ, psTTM, pcfNcfTTM,
        tradestatus, isST, pctChg
    """
    import baostock as bs
    import signal

    # 判断交易所前缀
    if stock_code.startswith('6'):
        bs_code = f'sh.{stock_code}'
    else:
        bs_code = f'sz.{stock_code}'

    # ---------- 设置超时闹钟 ----------
    # 用 signal.alarm 防止 baostock API 无限挂起（如 6/1 事故）
    class _Timeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise _Timeout(f'baostock API 超时 ({timeout_seconds}s)')

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_seconds)

    try:
        lg = bs.login()
        if lg.error_code != '0':
            print(f'[baostock] 登录失败: {lg.error_msg}')
            return None

        # 请求日线数据（前复权）
        fields = ('date,code,open,high,low,close,preclose,'
                  'volume,amount,adjustflag,turn,tradestatus,pctChg,isST')
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag='3',  # 前复权
        )

        if rs.error_code != '0':
            print(f'[baostock] 获取 {bs_code} 数据失败: {rs.error_msg}')
            bs.logout()
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        bs.logout()

    except _Timeout:
        print(f'[baostock] 获取 {bs_code} 数据超时 ({timeout_seconds}s)，跳过该股票')
        # 超时后不尝试 bs.logout()——它也可能卡住；让下一次 login() 重置连接
        return None
    finally:
        signal.alarm(0)               # 取消闹钟
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原 handler

    if not data_list:
        return None

    df = pd.DataFrame(data_list, columns=rs.fields)

    # ---- 清理与转换 ----

    # 去掉不需要的列
    for col in ['code', 'adjustflag']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 数值类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                    'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 整数类型
    if 'isST' in df.columns:
        df['isST'] = pd.to_numeric(df['isST'], errors='coerce').fillna(0).astype(int)
    else:
        df['isST'] = 0

    if 'tradestatus' in df.columns:
        df['tradestatus'] = pd.to_numeric(df['tradestatus'], errors='coerce').fillna(1).astype(int)
    else:
        df['tradestatus'] = 1

    # 日期转换
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 过滤非交易日（tradestatus == 0）
    if 'tradestatus' in df.columns:
        df = df[df['tradestatus'] == 1]

    # 过滤 ST 股
    if (df['isST'] == 1).any():
        return None

    # 补全本地 CSV 中可能存在但 baostock 不提供的列
    for col in ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']:
        if col not in df.columns:
            df[col] = float('nan')

    # 确保关键列无缺失
    required = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    df = df.dropna(subset=required)

    if len(df) == 0:
        return None

    return df.reset_index(drop=True)


def load_index_data(
    start_date: str = '',
    end_date: str = '',
) -> Optional[pd.DataFrame]:
    """
    加载沪深300指数日线数据。

    Parameters
    ----------
    start_date : str
        开始日期 'YYYY-MM-DD'，空=全部
    end_date : str
        结束日期 'YYYY-MM-DD'，空=到最新

    Returns
    -------
    pd.DataFrame or None
    """
    try:
        df = pd.read_csv(HS300_INDEX_FILE)
        df['date'] = pd.to_datetime(df['date'])

        if start_date:
            start = pd.Timestamp(start_date)
            df = df[df['date'] >= start]

        if end_date:
            end = pd.Timestamp(end_date)
            df = df[df['date'] <= end]

        # 计算累积收益
        if len(df) > 0:
            df['benchmark_returns'] = df['close'].pct_change()
            df['benchmark_cumulative_returns'] = (
                1 + df['benchmark_returns'].fillna(0)
            ).cumprod()

        return df if len(df) > 0 else None

    except FileNotFoundError:
        print(f'[load_index] 文件不存在: {HS300_INDEX_FILE}')
        return None
