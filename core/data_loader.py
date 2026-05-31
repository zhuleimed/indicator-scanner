"""
数据加载模块 (DataLoader)

职责：
  1. 从本地 CSV 目录加载个股日线数据
  2. 数据清洗与校验（ST 股过滤、次新股过滤）
  3. 日期范围裁剪
  4. 基准指数数据获取（baostock）
  5. 统一输出标准化的 DataFrame

输出列规范：
  date, open, high, low, close, volume, amount, pct_chg, cumulative_returns
"""

import os
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import baostock as bs

# baostock 不是线程安全的，benchmark 数据加载需全局锁
_benchmark_lock = threading.Lock()

# 项目根目录常量（相对于本模块的位置）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class DataLoader:
    """数据加载器：负责加载、清洗、裁剪股票日线数据"""

    # 输入文件必需列
    REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'isST']
    # 引擎使用的数值列
    NUMERIC_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']

    def __init__(self, data_dir: Optional[str] = None):
        """
        Parameters
        ----------
        data_dir : str, optional
            股票 CSV 文件所在目录。默认取项目的 ../data/input/
        """
        if data_dir is None:
            # 向上两级到 code/，再去 data/input/
            code_root = os.path.dirname(PROJECT_ROOT)
            parent = os.path.dirname(code_root)
            data_dir = os.path.join(parent, 'data', 'input')
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load_stock(self,
                   stock_code: str,
                   start_date: str,
                   end_date: str = '',
                   min_days: int = 240) -> pd.DataFrame:
        """
        加载单只股票并做标准化预处理。

        Returns
        -------
        pd.DataFrame
            包含列：date, open, high, low, close, volume, amount,
                   pct_chg, cumulative_returns, stock_code
        """
        # 1. 读取 CSV
        file_path = os.path.join(self.data_dir, f'{stock_code}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'股票数据文件不存在: {file_path}')

        df = pd.read_csv(file_path)
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f'数据文件缺少必要列: {col}')

        # 2. 日期转换
        df['date'] = pd.to_datetime(df['date'])

        # 3. 过滤 ST 股（最近交易日 isST == 1）
        if df.iloc[-1]['isST'] == 1:
            raise ValueError(f'股票 {stock_code} 当前为 ST 股，不符合要求')

        # 4. 过滤次新股（不足 min_days 个交易日）
        if len(df) < min_days:
            raise ValueError(f'股票 {stock_code} 为次新股（{len(df)}行 < {min_days}行）')

        # 5. 数值列类型转换
        for col in self.NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 6. 裁剪日期范围
        first_date, last_date = df['date'].min(), df['date'].max()
        start = pd.to_datetime(start_date)
        if start < first_date:
            raise ValueError(f'回测开始 {start_date} 早于数据起始 {first_date.date()}')
        if end_date:
            end = pd.to_datetime(end_date)
            if end > last_date:
                raise ValueError(f'回测结束 {end_date} 晚于数据截止 {last_date.date()}')
        else:
            end = last_date

        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        df = df.reset_index(drop=True)

        # 7. 选取引擎需要的列
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        # 8. 计算日收益率和累积收益率
        df['pct_chg'] = df['close'].pct_change()
        df['cumulative_returns'] = (1 + df['pct_chg']).cumprod()
        df.loc[0, 'cumulative_returns'] = 1.0

        df['stock_code'] = stock_code
        return df.dropna().reset_index(drop=True)

    def load_benchmark(self,
                       benchmark_code: str,
                       start_date: str,
                       end_date: str = '') -> pd.DataFrame:
        """
        从 baostock 获取基准指数日线数据，计算累积收益率。

        Returns
        -------
        pd.DataFrame
            包含列：date, open, high, low, close, benchmark_returns,
                   benchmark_cumulative_returns
        """
        start_obj = pd.to_datetime(start_date)
        # baostock 可能需要前缀数据用于 pct_change，提前 1 个月
        fetch_start = (start_obj - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d') if end_date else ''

        with _benchmark_lock:
            bs.login()
            rs = bs.query_history_k_data_plus(
                benchmark_code,
                'date,open,high,low,close',
                start_date=fetch_start,
                end_date=fetch_end,
                frequency='d',
            )
            records = []
            while rs.error_code == '0' and rs.next():
                records.append(rs.get_row_data())
            bs.logout()

        df = pd.DataFrame(records, columns=rs.fields)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_obj)].reset_index(drop=True)
        df = df.sort_values('date')

        df['benchmark_returns'] = df['close'].pct_change()
        df['benchmark_cumulative_returns'] = (1 + df['benchmark_returns']).cumprod()
        df.loc[0, 'benchmark_cumulative_returns'] = 1.0
        return df

    def load_stock_batch(self,
                         stock_codes: list,
                         start_date: str,
                         end_date: str = '',
                         min_days: int = 240,
                         on_error: str = 'skip') -> dict:
        """
        批量加载股票，返回 {stock_code: DataFrame} 字典。

        Parameters
        ----------
        on_error : str
            'skip' — 加载失败的股票自动跳过；
            'raise' — 遇到失败立即抛出。
        """
        result = {}
        for code in stock_codes:
            try:
                df = self.load_stock(code, start_date, end_date, min_days)
                result[code] = df
            except Exception as exc:
                if on_error == 'raise':
                    raise
                print(f'  ⚠ 跳过 {code}: {exc}')
        return result
