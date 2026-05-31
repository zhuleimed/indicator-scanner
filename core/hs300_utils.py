"""
沪深300工具模块 — 015_indicator_scanner

职责：
  1. 从 baostock 获取沪深300成分股列表
  2. 判断当前是否为交易日
  3. 加载指数日线数据
"""

from datetime import date, datetime
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

    策略：读取参考股票 CSV 最后一行的日期，如果等于今天的日期，
    说明今天数据已更新，即为交易日。

    Returns
    -------
    bool
    """
    today = date.today()
    ref_file = f'{REFERENCE_STOCK}.csv'
    ref_path = f'{DATA_DIR}/{ref_file}'

    try:
        df = pd.read_csv(ref_path)
        last_date_str = str(df['date'].iloc[-1])
        last_date = datetime.strptime(last_date_str[:10], '%Y-%m-%d').date()
        return last_date == today
    except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
        print(f'[交易日判断] 无法读取参考文件 {ref_path}: {e}')
        # 退而求其次：周末肯定不是交易日
        return today.weekday() < 5


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
