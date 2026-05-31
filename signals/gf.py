"""
GF 指标公式买卖信号 — 统一策略类

整合 GF_factors.py 中近 100 个技术指标计算函数和
GF_buy_sell_signal.py 中约 65 个买卖信号生成函数。

使用方法:
  python run_backtest.py --signal GF --stocks 000012 --indicator KDJ
  python run_backtest.py --signal GF --stocks 000012 --indicator MACD
  python run_compare.py --strategies GF-KDJ,GF-RSI,GF-MACD
"""

import numpy as np
import pandas as pd

from core.signal_engine import BaseSignal
from . import gf_factors as gf


class GFSignal(BaseSignal):
    """
    GF 综合信号策略（统一入口，通过 indicator 参数选择具体指标）。

    Parameters
    ----------
    indicator : str
        指标名称（大小写不敏感）。可选值见 INDICATORS 常量。
    **kwargs : dict
        各指标的特定参数，如 KDJ 的 N、MACD 的 N1/N2/N3 等。
    """

    name = 'GF'  # 信号引擎按此值查找 GF_signal 列

    # ======================================================================
    # 可用指标列表（约 65 个）
    # ======================================================================
    INDICATORS = [
        # ---- 价格动量指标（表34）----
        'DPO', 'ER', 'TII', 'PO', 'MA_DISPLACED', 'T3', 'POS',
        'PAC', 'ADTM', 'ZLMACD', 'TMA', 'TYP', 'KDJD', 'VMA',
        'BIAS', 'WMA_M', 'DDI', 'HMA', 'SROC', 'EXPMA', 'DC',
        'VIDYA', 'QSTICK', 'FB', 'DEMA', 'APZ', 'ASI', 'ARRON',
        'KC', 'MTM', 'CR', 'BOP', 'HULLMA', 'COPP', 'ENV',
        'RSIH', 'HLMA', 'TSI', 'BIAS36', 'UOS', 'DZRSI', 'DZCCI',
        'CMF', 'PPO', 'RWI', 'ATR', 'WAD', 'KST', 'VI',
        'DMA_I', 'MICD', 'PMO', 'RCCD', 'KAMA', 'AWS', 'ARBR',
        'ADXR', 'SMI', 'SI', 'DO', 'DBCD', 'CV',
        # ---- 价格反转指标（表35）----
        'KDJ', 'RMI', 'SKDJ', 'CCI', 'RSI', 'ROC', 'WR',
        'STC', 'RVI', 'RSIS',
        # ---- 成交量指标（表36）----
        'MAAMT', 'SROCVOL', 'PVO', 'BIASVOL', 'MACDVOL', 'ROCVOL',
        # ---- 价量指标（表37）----
        'VWAP', 'FI', 'NVI', 'PVT', 'RSIV', 'AMV', 'VRAMT',
        'WVAD', 'OBV', 'CMF_V', 'PVI', 'TMF', 'MFI', 'ADOSC',
        'VAO', 'VR', 'KO', 'EMV',
        # ---- 混合指标 ----
        'MACD',
    ]

    # ======================================================================
    # 策略参数：indicator 及其默认参数
    # ======================================================================

    _DEFAULT_PARAMS = {
        # ----- 价格动量（表34）-----
        'DPO':          {'N': 20},
        'ER':           {'N': 20},
        'TII':          {'N1': 40, 'M': 20, 'N2': 9},
        'PO':           {'N1': 9, 'N2': 26},
        'MA_DISPLACED': {'N': 20, 'M': 10},
        'T3':           {'N': 20, 'VA': 0.7},
        'POS':          {'N': 100},
        'PAC':          {'N1': 20, 'N2': 20},
        'ADTM':         {'N': 20},
        'ZLMACD':       {'N1': 20, 'N2': 100},
        'TMA':          {'N': 20},
        'TYP':          {'N1': 10, 'N2': 30},
        'KDJD':         {'N': 9, 'M': 3},
        'VMA':          {'N': 20},
        'BIAS':         {'N': 6},
        'WMA_M':        {'N': 20},
        'DDI':          {'N': 40},
        'HMA':          {'N': 20},
        'SROC':         {'N': 13, 'M': 21},
        'EXPMA':        {'N': 12},
        'DC':           {'N': 20},
        'VIDYA':        {'N': 10},
        'QSTICK':       {'N': 20},
        'FB':           {'N': 20, 'PARAM': 1.618},
        'DEMA':         {'N': 60},
        'APZ':          {'N': 10, 'M': 20, 'PARAM': 2},
        'ASI':          {'N': 20, 'M': 20},
        'ARRON':        {'N': 20},
        'KC':           {'N': 14, 'M': 20},
        'MTM':          {'N': 60},
        'CR':           {'N': 20},
        'BOP':          {'N': 20},
        'HULLMA':       {'N': 20},
        'COPP':         {'N1': 10, 'N2': 20, 'M': 5},
        'ENV':          {'N': 25, 'PARAM': 0.05},
        'RSIH':         {'N1': 40, 'N2': 20},
        'HLMA':         {'N1': 20, 'N2': 20},
        'TSI':          {'N1': 25, 'N2': 13},
        'BIAS36':       {'N': 6},
        'UOS':          {'N1': 7, 'N2': 14, 'N3': 28},
        'DZRSI':        {'N': 14, 'M': 3, 'PARAM': 2},
        'DZCCI':        {'N': 40, 'M': 3, 'PARAM': 2},
        'CMF':          {'N': 20},
        'PPO':          {'N1': 12, 'N2': 26, 'N3': 9},
        'RWI':          {'N': 14},
        'ATR':          {'N': 14},
        'WAD':          {'N': 20},
        'KST':          {'N1': 10, 'N2': 15, 'N3': 20, 'N4': 30, 'M': 9},
        'VI':           {'N': 40},
        'DMA_I':        {'N1': 10, 'N2': 50, 'M': 10},
        'MICD':         {'N': 20, 'N1': 10, 'N2': 20, 'M': 10},
        'PMO':          {'N1': 10, 'N2': 40, 'N3': 20},
        'RCCD':         {'N': 40, 'N1': 20, 'N2': 40, 'M': 40},
        'KAMA':         {'N': 10, 'N1': 2, 'N2': 30},
        'AWS':          {'N': 20},
        'ARBR':         {'N': 26},
        'ADXR':         {'N': 6},
        'SMI':          {'N1': 20, 'N2': 20, 'N3': 20},
        'SI':           {'N': 20, 'M': 20},
        'DO':           {'N': 20},
        'DBCD':         {'N': 5, 'M': 16, 'T': 17},
        'CV':           {'N': 10},
        # ----- 价格反转（表35）-----
        'KDJ':          {'N': 40},
        'RMI':          {'N': 7},
        'SKDJ':         {'N': 60, 'M': 5},
        'CCI':          {'N': 14},
        'RSI':          {'N': 24},
        'ROC':          {'N': 100},
        'WR':           {'N': 10},
        'STC':          {'N1': 23, 'N2': 50, 'N': 40},
        'RVI':          {'N1': 10, 'N2': 20},
        'RSIS':         {'N': 120, 'M': 20},
        # ----- 成交量（表36）-----
        'MAAMT':        {'N': 40},
        'SROCVOL':      {'N': 20, 'M': 10},
        'PVO':          {'N1': 12, 'N2': 26},
        'BIASVOL':      {'N': 6},
        'MACDVOL':      {'N1': 20, 'N2': 40, 'N3': 10},
        'ROCVOL':       {'N': 80},
        # ----- 价量（表37）-----
        'VWAP':         {'N': 20},
        'FI':           {'N': 13},
        'NVI':          {'N': 144},
        'PVT':          {'N1': 13, 'N2': 34},
        'RSIV':         {'N': 20},
        'AMV':          {'N1': 13, 'N2': 34},
        'VRAMT':        {'N': 40},
        'WVAD':         {'N': 20},
        'OBV':          {'N1': 10, 'N2': 30},
        'PVI':          {'N': 40},
        'TMF':          {'N': 20},
        'MFI':          {'N': 14},
        'ADOSC':        {'N1': 10, 'N2': 30},
        'VAO':          {'N1': 10, 'N2': 30},
        'VR':           {'N': 40},
        'KO':           {'N1': 34, 'N2': 55},
        'EMV':          {'N': 20},
        # ----- 混合指标 -----
        'MACD':         {'N1': 12, 'N2': 26, 'N3': 9},
    }

    # ======================================================================
    # 初始化
    # ======================================================================

    def __init__(self, indicator: str = 'KDJ', **kwargs):
        """
        Parameters
        ----------
        indicator : str
            指标名称，不区分大小写。默认 KDJ。
        **kwargs : dict
            覆盖各指标的默认参数。
        """
        # 统一指标名称（转大写，去下划线和空格）
        key = indicator.upper().replace('_', '').replace(' ', '')
        match = [k for k in self.INDICATORS if k.replace('_', '') == key]
        if not match:
            candidates = ', '.join(self.INDICATORS)
            raise ValueError(
                f'未知指标: {indicator}\n可选指标({len(self.INDICATORS)}个):\n  {candidates}'
            )
        self._indicator = match[0]

        # 合并默认参数和用户指定参数
        defaults = self._DEFAULT_PARAMS.get(self._indicator, {}).copy()
        defaults.update(kwargs)
        self.params = defaults
        # 记录指标名以便信号列命名
        self._indicator_label = self._indicator.lower().replace(' ', '_')

    # ======================================================================
    # compute() — 唯一需要实现的方法
    # ======================================================================

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据选择的 indicator 生成买卖信号。
        信号列名: {indicator}_signal
        """
        df = data.copy()
        method_name = f'_signal_{self._indicator}'
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(
                f'指标 {self._indicator} 的信号方法未实现'
            )
        df = method(df)

        # 将生成的具体指标信号列（如 KDJ_signal）重命名为 GF_signal，
        # 这样才能被 SignalEngine 正确识别（引擎按 strategy.name + '_signal' 查找）
        old_col = f'{self._indicator}_signal'
        if old_col in df.columns:
            df['GF_signal'] = df[old_col]
            df.drop(columns=[old_col], inplace=True)

        return df

    # ======================================================================
    # 信号生成方法（每个指标一个）
    # 统一的信号逻辑模式：
    #   - 绝大多数用 CROSSOVER（上穿/下穿）
    #   - 少数用 threshold（阈值突破）
    #   - 极少数用 band（轨道突破）
    # ======================================================================

    # ------------------- 辅助工具 -------------------

    @staticmethod
    def _cross_up(series, threshold):
        """series 上穿 threshold 的位置"""
        s = pd.Series(series)
        return (s.shift(1) < threshold) & (s >= threshold)

    @staticmethod
    def _cross_down(series, threshold):
        """series 下穿 threshold 的位置"""
        s = pd.Series(series)
        return (s.shift(1) > threshold) & (s <= threshold)

    @staticmethod
    def _cross_over(s1, s2):
        """s1 上穿 s2"""
        s1, s2 = pd.Series(s1), pd.Series(s2)
        return (s1.shift(1) < s2.shift(1)) & (s1 >= s2)

    @staticmethod
    def _cross_under(s1, s2):
        """s1 下穿 s2"""
        s1, s2 = pd.Series(s1), pd.Series(s2)
        return (s1.shift(1) > s2.shift(1)) & (s1 <= s2)

    # ==================== 价格动量指标（表34）====================

    def _signal_DPO(self, df):
        n = self.params['N']
        val = gf.DPO(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['DPO_signal'] = 0
        df.loc[self._cross_up(s, 0), 'DPO_signal'] = 1
        df.loc[self._cross_down(s, 0), 'DPO_signal'] = -1
        return df

    def _signal_ER(self, df):
        n = self.params['N']
        bp, bp_ = gf.ER(df['close'].values, df['high'].values,
                         df['low'].values, n)
        bear = pd.Series(bp_, index=df.index)
        bull = pd.Series(bp, index=df.index)
        df['ER_signal'] = 0
        df.loc[self._cross_up(bear, 0), 'ER_signal'] = 1
        df.loc[self._cross_down(bull, 0), 'ER_signal'] = -1
        return df

    def _signal_TII(self, df):
        p = self.params
        val = gf.TII(df['close'].values, p['N1'], p['M'], p['N2'])
        sig = gf.EMA(val, p['N2'])
        s, sig = pd.Series(val, index=df.index), pd.Series(sig, index=df.index)
        df['TII_signal'] = 0
        df.loc[self._cross_over(s, sig), 'TII_signal'] = 1
        df.loc[self._cross_under(s, sig), 'TII_signal'] = -1
        return df

    def _signal_PO(self, df):
        p = self.params
        val = gf.PO(df['close'].values, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['PO_signal'] = 0
        df.loc[self._cross_up(s, 0), 'PO_signal'] = 1
        df.loc[self._cross_down(s, 0), 'PO_signal'] = -1
        return df

    def _signal_MA_DISPLACED(self, df):
        p = self.params
        val = gf.MADisplaced(df['close'].values, p['N'], p['M'])
        s = pd.Series(val, index=df.index)
        df['MA_DISPLACED_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'MA_DISPLACED_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'MA_DISPLACED_signal'] = -1
        return df

    def _signal_T3(self, df):
        p = self.params
        val = gf.T3(df['close'].values, p['N'], p['VA'])
        s = pd.Series(val, index=df.index)
        df['T3_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'T3_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'T3_signal'] = -1
        return df

    def _signal_POS(self, df):
        n = self.params['N']
        val = gf.POS(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['POS_signal'] = 0
        df.loc[self._cross_up(s, 80), 'POS_signal'] = 1
        df.loc[self._cross_down(s, 20), 'POS_signal'] = -1
        return df

    def _signal_PAC(self, df):
        p = self.params
        upper, lower = gf.PAC(df['high'].values, df['low'].values, p['N1'], p['N2'])
        df['PAC_signal'] = 0
        df.loc[df['close'] > upper, 'PAC_signal'] = 1
        df.loc[df['close'] < lower, 'PAC_signal'] = -1
        return df

    def _signal_ADTM(self, df):
        n = self.params['N']
        val = gf.ADTM(df['open'].values, df['high'].values, df['low'].values, n)
        s = pd.Series(val, index=df.index)
        df['ADTM_signal'] = 0
        df.loc[self._cross_up(s, 0.5), 'ADTM_signal'] = 1
        df.loc[self._cross_down(s, -0.5), 'ADTM_signal'] = -1
        return df

    def _signal_ZLMACD(self, df):
        p = self.params
        val = gf.ZLMACD(df['close'].values, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['ZLMACD_signal'] = 0
        df.loc[self._cross_up(s, 0), 'ZLMACD_signal'] = 1
        df.loc[self._cross_down(s, 0), 'ZLMACD_signal'] = -1
        return df

    def _signal_TMA(self, df):
        n = self.params['N']
        val = gf.TMA(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['TMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'TMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'TMA_signal'] = -1
        return df

    def _signal_TYP(self, df):
        p = self.params
        t1, t2 = gf.TYP(df['close'].values, df['high'].values,
                         df['low'].values, p['N1'], p['N2'])
        s1, s2 = pd.Series(t1, index=df.index), pd.Series(t2, index=df.index)
        df['TYP_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'TYP_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'TYP_signal'] = -1
        return df

    def _signal_KDJD(self, df):
        p = self.params
        k, d = gf.KDJD(df['close'].values, df['high'].values,
                        df['low'].values, p['N'], p['M'])
        sd = pd.Series(d, index=df.index)
        df['KDJD_signal'] = 0
        df.loc[self._cross_up(sd, 70), 'KDJD_signal'] = 1
        df.loc[self._cross_down(sd, 30), 'KDJD_signal'] = -1
        return df

    def _signal_VMA(self, df):
        n = self.params['N']
        val = gf.VMA(df['close'].values, df['high'].values,
                     df['low'].values, df['open'].values, n)
        s = pd.Series(val, index=df.index)
        df['VMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'VMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'VMA_signal'] = -1
        return df

    def _signal_BIAS(self, df):
        n = self.params['N']
        b6 = gf.BIAS(df['close'].values, 6)
        b12 = gf.BIAS(df['close'].values, 12)
        b24 = gf.BIAS(df['close'].values, 24)
        df['BIAS_signal'] = 0
        df.loc[(b6 > 5) & (b12 > 7) & (b24 > 11), 'BIAS_signal'] = 1
        df.loc[(b6 < 5) & (b12 < 7) & (b24 < 11), 'BIAS_signal'] = -1
        return df

    # ==================== 价格反转指标（表35）====================

    def _signal_KDJ(self, df):
        n = self.params['N']
        low_n = gf.LLV(df['low'].values, n)
        high_n = gf.HHV(df['high'].values, n)
        rsv = (df['close'] - low_n) / (high_n - low_n) * 100
        rsv_s = pd.Series(rsv, index=df.index)
        k = rsv_s.rolling(3).mean()
        d = k.rolling(3).mean()
        df['KDJ_signal'] = 0
        buy = (d < 20) & (k.shift(1) < d.shift(1)) & (k >= d)
        sell = (d > 80) & (k.shift(1) > d.shift(1)) & (k <= d)
        df.loc[buy, 'KDJ_signal'] = 1
        df.loc[sell, 'KDJ_signal'] = -1
        return df

    def _signal_RMI(self, df):
        n = self.params['N']
        val = gf.RMI(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['RMI_signal'] = 0
        df.loc[self._cross_up(s, 70), 'RMI_signal'] = 1
        df.loc[self._cross_down(s, 30), 'RMI_signal'] = -1
        return df

    def _signal_CCI(self, df):
        n = self.params['N']
        val = gf.CCI(df['close'].values, df['high'].values,
                     df['low'].values, n)
        s = pd.Series(val, index=df.index)
        df['CCI_signal'] = 0
        df.loc[self._cross_down(s, -100), 'CCI_signal'] = 1
        df.loc[self._cross_up(s, 100), 'CCI_signal'] = -1
        return df

    def _signal_RSI(self, df):
        n = self.params['N']
        val = gf.RSI(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['RSI_signal'] = 0
        df.loc[self._cross_up(s, 40), 'RSI_signal'] = 1
        df.loc[self._cross_down(s, 60), 'RSI_signal'] = -1
        return df

    def _signal_ROC(self, df):
        n = self.params['N']
        val = gf.ROC(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['ROC_signal'] = 0
        df.loc[self._cross_up(s, 5), 'ROC_signal'] = 1
        df.loc[self._cross_down(s, -5), 'ROC_signal'] = -1
        return df

    def _signal_WR(self, df):
        n = self.params['N']
        hn = gf.HHV(df['high'].values, n)
        ln = gf.LLV(df['low'].values, n)
        wr = 100 * (hn - df['close']) / (hn - ln)
        s = pd.Series(wr, index=df.index)
        df['WR_signal'] = 0
        df.loc[self._cross_up(s, 80), 'WR_signal'] = 1
        df.loc[self._cross_down(s, 20), 'WR_signal'] = -1
        return df

    # ==================== 价量指标（表37）====================

    def _signal_OBV(self, df):
        p = self.params
        close = df['close'].values
        vol = df['volume'].values
        obv = gf.CUM_SUM(np.where(close > gf.REF(close, 1), vol,
                                  np.where(close < gf.REF(close, 1), -vol, 0)))
        ma1 = gf.EMA(obv, p['N1'])
        ma2 = gf.EMA(obv, p['N2'])
        s1, s2 = pd.Series(ma1, index=df.index), pd.Series(ma2, index=df.index)
        df['OBV_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'OBV_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'OBV_signal'] = -1
        return df

    def _signal_MFI(self, df):
        n = self.params['N']
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        mf_pos = gf.SUM(np.where(tp >= gf.REF(tp, 1), mf, 0), n)
        mf_neg = gf.SUM(np.where(tp <= gf.REF(tp, 1), mf, 0), n)
        mf_neg = np.where(mf_neg == 0, 0.0001, mf_neg)
        val = 100 - 100 / (1 + mf_pos / mf_neg)
        s = pd.Series(val, index=df.index)
        df['MFI_signal'] = 0
        df.loc[self._cross_up(s, 80), 'MFI_signal'] = 1
        df.loc[self._cross_down(s, 20), 'MFI_signal'] = -1
        return df

    def _signal_VWAP(self, df):
        n = self.params['N']
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        val = gf.SUM(mf, n) / gf.SUM(df['volume'].values, n)
        s = pd.Series(val, index=df.index)
        df['VWAP_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'VWAP_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'VWAP_signal'] = -1
        return df

    # ==================== 混合指标 ====================

    def _signal_MACD(self, df):
        p = self.params
        ema12 = gf.EMA(df['close'].values, p['N1'])
        ema26 = gf.EMA(df['close'].values, p['N2'])
        dif = ema12 - ema26
        dea = gf.EMA(dif, p['N3'])
        sdif = pd.Series(dif, index=df.index)
        sdea = pd.Series(dea, index=df.index)
        df['MACD_signal'] = 0
        df.loc[self._cross_over(sdif, sdea), 'MACD_signal'] = 1
        df.loc[self._cross_under(sdif, sdea), 'MACD_signal'] = -1
        return df

    # ==================== 剩余通用指标模板 ====================
    # 以下指标都使用相同的 cross-over 模式：
    #   计算指标值 → 上穿0买入 / 下穿0卖出

    _CROSS_ZERO = [
        'SROC', 'EXPMA', 'QSTICK', 'MTM', 'FI', 'PVO',
        'ROCVOL', 'TMF', 'KO', 'EMV', 'MICD', 'RCCD',
    ]

    _CROSS_MA = [
        'WMA_M', 'HMA', 'DC', 'VIDYA', 'DEMA', 'HULLMA',
        'COPP', 'KAMA', 'AWS', 'MAAMT', 'VWAP',
    ]

    def _signal_SROC(self, df):
        p = self.params
        val = gf.SROC(df['close'].values, p['N'], p['M'])
        s = pd.Series(val, index=df.index)
        df['SROC_signal'] = 0
        df.loc[self._cross_up(s, 0), 'SROC_signal'] = 1
        df.loc[self._cross_down(s, 0), 'SROC_signal'] = -1
        return df

    def _signal_EXPMA(self, df):
        n = self.params['N']
        val = gf.EMA(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['EXPMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'EXPMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'EXPMA_signal'] = -1
        return df

    def _signal_QSTICK(self, df):
        n = self.params['N']
        val = gf.Qstick(df['close'].values, df['open'].values, n)
        s = pd.Series(val, index=df.index)
        df['QSTICK_signal'] = 0
        df.loc[self._cross_up(s, 0), 'QSTICK_signal'] = 1
        df.loc[self._cross_down(s, 0), 'QSTICK_signal'] = -1
        return df

    def _signal_MTM(self, df):
        n = self.params['N']
        val = gf.MTM(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['MTM_signal'] = 0
        df.loc[self._cross_up(s, 0), 'MTM_signal'] = 1
        df.loc[self._cross_down(s, 0), 'MTM_signal'] = -1
        return df

    def _signal_FI(self, df):
        n = self.params['N']
        val = (df['close'] - gf.REF(df['close'].values, 1)) * df['volume']
        fima = gf.EMA(val, n)
        s = pd.Series(fima, index=df.index)
        df['FI_signal'] = 0
        df.loc[self._cross_up(s, 0), 'FI_signal'] = 1
        df.loc[self._cross_down(s, 0), 'FI_signal'] = -1
        return df

    def _signal_PVO(self, df):
        p = self.params
        ema1 = gf.EMA(df['volume'].values, p['N1'])
        ema2 = gf.EMA(df['volume'].values, p['N2'])
        val = (ema1 - ema2) / ema2
        s = pd.Series(val, index=df.index)
        df['PVO_signal'] = 0
        df.loc[self._cross_up(s, 0), 'PVO_signal'] = 1
        df.loc[self._cross_down(s, 0), 'PVO_signal'] = -1
        return df

    def _signal_MICD(self, df):
        p = self.params
        val = gf.MICD(df['close'].values, p['N'], p['N1'], p['N2'], p['M'])
        s = pd.Series(val, index=df.index)
        df['MICD_signal'] = 0
        df.loc[self._cross_up(s, 0), 'MICD_signal'] = 1
        df.loc[self._cross_down(s, 0), 'MICD_signal'] = -1
        return df

    def _signal_RCCD(self, df):
        p = self.params
        val = gf.RCCD(df['close'].values, p['N'], p['N1'], p['N2'], p['M'])
        s = pd.Series(val, index=df.index)
        df['RCCD_signal'] = 0
        df.loc[self._cross_up(s, 0), 'RCCD_signal'] = 1
        df.loc[self._cross_down(s, 0), 'RCCD_signal'] = -1
        return df

    def _signal_KAMA(self, df):
        raw = gf.KAMA(df['close'].values)
        s = pd.Series(raw, index=df.index)
        df['KAMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'KAMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'KAMA_signal'] = -1
        return df

    def _signal_AWS(self, df):
        n = self.params['N']
        val = gf.AWS(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['AWS_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'AWS_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'AWS_signal'] = -1
        return df

    # ==================== 成交量/价量剩余指标 ====================

    def _signal_WVAD(self, df):
        n = self.params['N']
        val = gf.SUM(
            ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10))
            * df['volume'], n)
        s = pd.Series(val, index=df.index)
        df['WVAD_signal'] = 0
        df.loc[self._cross_up(s, 0), 'WVAD_signal'] = 1
        df.loc[self._cross_down(s, 0), 'WVAD_signal'] = -1
        return df

    def _signal_EMV(self, df):
        n = self.params['N']
        mp = (df['high'] + df['low']) / 2
        mpm = mp - gf.REF(mp, 1)
        br = df['volume'] / 1000000 / (df['high'] - df['low'] + 1e-10)
        val = gf.MA(mpm / br, n)
        s = pd.Series(val, index=df.index)
        df['EMV_signal'] = 0
        df.loc[self._cross_up(s, 0), 'EMV_signal'] = 1
        df.loc[self._cross_down(s, 0), 'EMV_signal'] = -1
        return df

    def _signal_VR(self, df):
        n = self.params['N']
        c, v = df['close'].values, df['volume'].values
        av = gf.SUM(np.where(c > gf.REF(c, 1), v, 0), n)
        bv = gf.SUM(np.where(c < gf.REF(c, 1), v, 0), n)
        cv_ = gf.SUM(np.where(c == gf.REF(c, 1), v, 0), n)
        val = (av + cv_ / 2) / (bv + cv_ / 2)
        s = pd.Series(val, index=df.index)
        df['VR_signal'] = 0
        df.loc[self._cross_up(s, 250), 'VR_signal'] = 1
        df.loc[self._cross_down(s, 300), 'VR_signal'] = -1
        return df

    # ==================== 未实现的指标（使用 N/A 信号，保持完整性）====================
    # 以下指标要么需要额外参数，要么信号逻辑复杂，以 CROSS_ZERO 兜底

    def _signal_DDI(self, df):
        n = self.params['N']
        val = gf.DDI(df['close'].values, df['high'].values, df['low'].values, n)
        s = pd.Series(val, index=df.index)
        df['DDI_signal'] = 0
        df.loc[self._cross_up(s, 0), 'DDI_signal'] = 1
        df.loc[self._cross_down(s, 0), 'DDI_signal'] = -1
        return df

    def _signal_CR(self, df):
        n = self.params['N']
        val = gf.CR(df['close'].values, df['high'].values, df['low'].values, n)
        s = pd.Series(val, index=df.index)
        df['CR_signal'] = 0
        df.loc[self._cross_up(s, 200), 'CR_signal'] = 1
        df.loc[self._cross_down(s, 50), 'CR_signal'] = -1
        return df

    def _signal_BOP(self, df):
        n = self.params['N']
        val = gf.BOP(df['close'].values, df['open'].values,
                     df['high'].values, df['low'].values, n)
        s = pd.Series(val, index=df.index)
        df['BOP_signal'] = 0
        df.loc[self._cross_up(s, 0.5), 'BOP_signal'] = 1
        df.loc[self._cross_down(s, -0.5), 'BOP_signal'] = -1
        return df

    def _signal_SKDJ(self, df):
        p = self.params
        low_n = gf.LLV(df['low'].values, p['N'])
        high_n = gf.HHV(df['high'].values, p['N'])
        rsv = (df['close'] - low_n) / (high_n - low_n) * 100
        rsv_s = pd.Series(rsv, index=df.index)
        k = rsv_s.rolling(3).mean().rolling(3).mean()
        d = k.rolling(3).mean()
        df['SKDJ_signal'] = 0
        buy = (d < 40) & (k.shift(1) < d.shift(1)) & (k >= d)
        sell = (d > 60) & (k.shift(1) > d.shift(1)) & (k <= d)
        df.loc[buy, 'SKDJ_signal'] = 1
        df.loc[sell, 'SKDJ_signal'] = -1
        return df


# ======================================================================
# 价格动量指标（补充）
# 模式: CROSS_ZERO = 上穿0买入/下穿0卖出
#       CROSS_THRESHOLD = 固定阈值穿越
#       CROSS_MA = 价格上穿均线买入/下穿卖出
#       CROSS_LINE = 两线交叉
# ======================================================================

# ------------------- CROSS_ZERO: 上穿0买入/下穿0卖出 -------------------

    def _signal_TSI(self, df):
        p = self.params
        val = gf.TSI(df['close'].values, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['TSI_signal'] = 0
        df.loc[self._cross_up(s, 0), 'TSI_signal'] = 1
        df.loc[self._cross_down(s, 0), 'TSI_signal'] = -1
        return df
    
    def _signal_PPO(self, df):
        p = self.params
        ppo_val, ppo_signal = gf.PPO(df['close'].values, p['N1'], p['N2'], p['N3'])
        s1, s2 = pd.Series(ppo_val, index=df.index), pd.Series(ppo_signal, index=df.index)
        df['PPO_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'PPO_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'PPO_signal'] = -1
        return df
    
    def _signal_PMO(self, df):
        p = self.params
        pmo_val, pmo_signal = gf.PMO(df['close'].values, p['N1'], p['N2'], p['N3'])
        s1, s2 = pd.Series(pmo_val, index=df.index), pd.Series(pmo_signal, index=df.index)
        df['PMO_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'PMO_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'PMO_signal'] = -1
        return df
    
    def _signal_KST(self, df):
        p = self.params
        val = gf.KST(df['close'].values, p['N1'], p['N2'], p['N3'], p['N4'], p['M'])
        s = pd.Series(val, index=df.index)
        df['KST_signal'] = 0
        df.loc[self._cross_up(s, 0), 'KST_signal'] = 1
        df.loc[self._cross_down(s, 0), 'KST_signal'] = -1
        return df
    
    def _signal_VI(self, df):
        p = self.params
        vi_val, vi_signal = gf.VI(df['close'].values, df['high'].values,
                                   df['low'].values, p['N'])
        s1, s2 = pd.Series(vi_val, index=df.index), pd.Series(vi_signal, index=df.index)
        df['VI_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'VI_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'VI_signal'] = -1
        return df
    
    def _signal_DMA_I(self, df):
        p = self.params
        ma_short = gf.MA(df['close'].values, p['N1'])
        ma_long = gf.MA(df['close'].values, p['N2'])
        dma_val = ma_short - ma_long
        ama = gf.MA(dma_val, p['M'])
        s1, s2 = pd.Series(dma_val, index=df.index), pd.Series(ama, index=df.index)
        df['DMA_I_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'DMA_I_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'DMA_I_signal'] = -1
        return df
    
    def _signal_ADOSC(self, df):
        p = self.params
        close, high, low, volume = (df[c].values for c in ('close', 'high', 'low', 'volume'))
        # 计算 Accumulation Distribution
        mfm = ((close - low) - (high - close)) / np.where(high - low > 0, high - low, 1e-10)
        ad_val = gf.CUM_SUM(mfm * volume)
        val = gf.ADOSC(ad_val, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['ADOSC_signal'] = 0
        df.loc[self._cross_up(s, 0), 'ADOSC_signal'] = 1
        df.loc[self._cross_down(s, 0), 'ADOSC_signal'] = -1
        return df
    
    def _signal_VAO(self, df):
        p = self.params
        vao1, vao2 = gf.VAO(df['close'].values, df['high'].values, df['low'].values,
                             df['volume'].values, p['N1'], p['N2'])
        s1, s2 = pd.Series(vao1, index=df.index), pd.Series(vao2, index=df.index)
        df['VAO_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'VAO_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'VAO_signal'] = -1
        return df
    
    def _signal_KO(self, df):
        p = self.params
        val = gf.KO(df['close'].values, df['high'].values, df['low'].values,
                    df['volume'].values, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['KO_signal'] = 0
        df.loc[self._cross_up(s, 0), 'KO_signal'] = 1
        df.loc[self._cross_down(s, 0), 'KO_signal'] = -1
        return df
    
    def _signal_NVI(self, df):
        p = self.params
        val = gf.NVI(df['close'].values, df['volume'].values, p['N'])
        s = pd.Series(val, index=df.index)
        df['NVI_signal'] = 0
        df.loc[self._cross_up(s, 0), 'NVI_signal'] = 1
        df.loc[self._cross_down(s, 0), 'NVI_signal'] = -1
        return df
    
    def _signal_PVT(self, df):
        p = self.params
        pvt, pvt_ma = gf.PVT(df['close'].values, df['volume'].values, p['N1'], p['N2'])
        s1, s2 = pd.Series(pvt, index=df.index), pd.Series(pvt_ma, index=df.index)
        df['PVT_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'PVT_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'PVT_signal'] = -1
        return df
    
    def _signal_TMF(self, df):
        p = self.params
        val = gf.TMF(df['close'].values, df['high'].values, df['low'].values,
                     df['volume'].values, p['N'])
        s = pd.Series(val, index=df.index)
        df['TMF_signal'] = 0
        df.loc[self._cross_up(s, 0), 'TMF_signal'] = 1
        df.loc[self._cross_down(s, 0), 'TMF_signal'] = -1
        return df
    
    def _signal_SI(self, df):
        p = self.params
        asi, asima = gf.SI(df['close'].values, df['high'].values, df['low'].values,
                           df['open'].values, p['N'], p['M'])
        s1, s2 = pd.Series(asi, index=df.index), pd.Series(asima, index=df.index)
        df['SI_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'SI_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'SI_signal'] = -1
        return df
    
    def _signal_DO(self, df):
        n = self.params['N']
        val = gf.DO(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['DO_signal'] = 0
        df.loc[self._cross_up(s, 0.5), 'DO_signal'] = 1
        df.loc[self._cross_down(s, 0.5), 'DO_signal'] = -1
        return df
    
    def _signal_DBCD(self, df):
        p = self.params
        val = gf.DBCD(df['close'].values, p['N'], p['M'], p['T'])
        s = pd.Series(val, index=df.index)
        df['DBCD_signal'] = 0
        df.loc[self._cross_up(s, 0), 'DBCD_signal'] = 1
        df.loc[self._cross_down(s, 0), 'DBCD_signal'] = -1
        return df
    
    def _signal_DZRSI(self, df):
        p = self.params
        upper, lower = gf.DZRSI(df['close'].values, p['N'], p['M'], p['PARAM'])
        rsi = (pd.Series(upper, index=df.index) + pd.Series(lower, index=df.index)) / 2
        df['DZRSI_signal'] = 0
        df.loc[self._cross_up(rsi, 40), 'DZRSI_signal'] = 1
        df.loc[self._cross_down(rsi, 60), 'DZRSI_signal'] = -1
        return df
    
    def _signal_DZCCI(self, df):
        p = self.params
        up, dn = gf.DZCCI(df['close'].values, df['high'].values, df['low'].values,
                           p['N'], p['M'], p['PARAM'])
        s1, s2 = pd.Series(up, index=df.index), pd.Series(dn, index=df.index)
        df['DZCCI_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'DZCCI_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'DZCCI_signal'] = -1
        return df
    
    # ------------------- CROSS_THRESHOLD: 固定阈值穿越 -------------------
    
    def _signal_UOS(self, df):
        p = self.params
        val = gf.UOS(df['close'].values, df['high'].values, df['low'].values,
                     p['N1'], p['N2'], p['N3'])
        s = pd.Series(val, index=df.index)
        df['UOS_signal'] = 0
        df.loc[self._cross_up(s, 30), 'UOS_signal'] = 1
        df.loc[self._cross_down(s, 70), 'UOS_signal'] = -1
        return df
    
    def _signal_RWI(self, df):
        p = self.params
        rwih, rwil = gf.RWI(df['close'].values, df['high'].values, df['low'].values, p['N'])
        s1, s2 = pd.Series(rwih, index=df.index), pd.Series(rwil, index=df.index)
        df['RWI_signal'] = 0
        df.loc[self._cross_up(s1, 1), 'RWI_signal'] = 1
        df.loc[self._cross_down(s2, 1), 'RWI_signal'] = -1
        return df
    
    def _signal_SMI(self, df):
        p = self.params
        smi_val, smi_signal = gf.SMI(df['close'].values, df['high'].values,
                                       df['low'].values,
                                       p['N1'], p['N2'], p['N3'])
        s1, s2 = pd.Series(smi_val, index=df.index), pd.Series(smi_signal, index=df.index)
        df['SMI_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'SMI_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'SMI_signal'] = -1
        return df
    
    def _signal_STC(self, df):
        p = self.params
        val = gf.STC(df['close'].values, df['high'].values, df['low'].values,
                     p['N1'], p['N2'], p['N'])
        s = pd.Series(val, index=df.index)
        df['STC_signal'] = 0
        df.loc[self._cross_up(s, 25), 'STC_signal'] = 1
        df.loc[self._cross_down(s, 75), 'STC_signal'] = -1
        return df
    
    def _signal_RVI(self, df):
        p = self.params
        val = gf.RVI(df['close'].values, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['RVI_signal'] = 0
        df.loc[self._cross_up(s, 40), 'RVI_signal'] = 1
        df.loc[self._cross_down(s, 60), 'RVI_signal'] = -1
        return df
    
    def _signal_RSIS(self, df):
        p = self.params
        val = gf.RSIS(df['close'].values, p['N'], p['M'])
        s = pd.Series(val, index=df.index)
        df['RSIS_signal'] = 0
        df.loc[self._cross_up(s, 40), 'RSIS_signal'] = 1
        df.loc[self._cross_down(s, 60), 'RSIS_signal'] = -1
        return df
    
    def _signal_RSIH(self, df):
        p = self.params
        val = gf.RSIH(df['close'].values, p['N1'], p['N2'])
        s = pd.Series(val, index=df.index)
        df['RSIH_signal'] = 0
        df.loc[self._cross_up(s, 40), 'RSIH_signal'] = 1
        df.loc[self._cross_down(s, 60), 'RSIH_signal'] = -1
        return df
    
    def _signal_BIAS36(self, df):
        n = self.params['N']
        val = gf.BIAS36(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['BIAS36_signal'] = 0
        df.loc[self._cross_up(s, 0), 'BIAS36_signal'] = 1
        df.loc[self._cross_down(s, 0), 'BIAS36_signal'] = -1
        return df
    
    def _signal_CV(self, df):
        p = self.params
        val = gf.CV(df['close'].values, df['high'].values, df['low'].values, p['N'])
        s = pd.Series(val, index=df.index)
        df['CV_signal'] = 0
        df.loc[self._cross_up(s, 0), 'CV_signal'] = 1
        df.loc[self._cross_down(s, 0), 'CV_signal'] = -1
        return df
    
    def _signal_ARBR(self, df):
        p = self.params
        ar, br = gf.ARBR(df['close'].values, df['high'].values, df['low'].values,
                         df['open'].values, p['N'])
        sar = pd.Series(ar, index=df.index)
        df['ARBR_signal'] = 0
        df.loc[self._cross_up(sar, 80), 'ARBR_signal'] = 1
        df.loc[self._cross_down(sar, 120), 'ARBR_signal'] = -1
        return df
    
    def _signal_ADXR(self, df):
        n = self.params['N']
        # 计算 ADX (Average Directional Index)
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        # TR (True Range)
        prev_close = gf.REF(close, 1)
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        # +DM, -DM
        up_move = high - gf.REF(high, 1)
        down_move = gf.REF(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        # Smooth
        atr = gf.MA(tr, n)
        di_plus = 100 * gf.MA(plus_dm, n) / np.where(atr > 0, atr, 1e-10)
        di_minus = 100 * gf.MA(minus_dm, n) / np.where(atr > 0, atr, 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / np.where(di_plus + di_minus > 0, di_plus + di_minus, 1e-10)
        adx = gf.MA(dx, n)
        adxr = gf.ADXR(adx, n)
        s = pd.Series(adxr, index=df.index)
        df['ADXR_signal'] = 0
        df.loc[self._cross_up(s, 25), 'ADXR_signal'] = 1
        df.loc[self._cross_down(s, 20), 'ADXR_signal'] = -1
        return df
    
    # ------------------- CROSS_MA: 价格上穿/下穿均线 -------------------
    
    def _signal_WMA_M(self, df):
        n = self.params['N']
        val = gf.WMA(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['WMA_M_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'WMA_M_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'WMA_M_signal'] = -1
        return df
    
    def _signal_HMA(self, df):
        n = self.params['N']
        half_n = max(n // 2, 1)
        wma_half = gf.WMA(df['close'].values, half_n)
        wma_full = gf.WMA(df['close'].values, n)
        hull = gf.WMA(2 * wma_half - wma_full, int(max(np.sqrt(n), 1)))
        s = pd.Series(hull, index=df.index)
        df['HMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'HMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'HMA_signal'] = -1
        return df
    
    def _signal_HULLMA(self, df):
        n = self.params['N']
        val = gf.HULLMA(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['HULLMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'HULLMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'HULLMA_signal'] = -1
        return df
    
    def _signal_DC(self, df):
        n = self.params['N']
        upper, lower, mid = gf.DC(df['high'].values, df['low'].values, n)
        s_upper = pd.Series(upper, index=df.index)
        s_lower = pd.Series(lower, index=df.index)
        df['DC_signal'] = 0
        df.loc[df['close'] > s_upper, 'DC_signal'] = 1
        df.loc[df['close'] < s_lower, 'DC_signal'] = -1
        return df
    
    def _signal_VIDYA(self, df):
        n = self.params['N']
        val = gf.VIDYA(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['VIDYA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'VIDYA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'VIDYA_signal'] = -1
        return df
    
    def _signal_DEMA(self, df):
        n = self.params['N']
        val = gf.DEMA(df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['DEMA_signal'] = 0
        df.loc[self._cross_over(df['close'], s), 'DEMA_signal'] = 1
        df.loc[self._cross_under(df['close'], s), 'DEMA_signal'] = -1
        return df
    
    def _signal_COPP(self, df):
        p = self.params
        val = gf.COPP(df['close'].values, p['N1'], p['N2'], p['M'])
        s = pd.Series(val, index=df.index)
        df['COPP_signal'] = 0
        df.loc[self._cross_up(s, 0), 'COPP_signal'] = 1
        df.loc[self._cross_down(s, 0), 'COPP_signal'] = -1
        return df
    
    def _signal_ENV(self, df):
        p = self.params
        upper, lower = gf.ENV(df['close'].values, p['N'], p['PARAM'])
        s_upper = pd.Series(upper, index=df.index)
        s_lower = pd.Series(lower, index=df.index)
        df['ENV_signal'] = 0
        df.loc[df['close'] > s_upper, 'ENV_signal'] = 1
        df.loc[df['close'] < s_lower, 'ENV_signal'] = -1
        return df
    
    def _signal_HLMA(self, df):
        p = self.params
        hma, lma = gf.HLMA(df['high'].values, df['low'].values, p['N1'], p['N2'])
        s1, s2 = pd.Series(hma, index=df.index), pd.Series(lma, index=df.index)
        df['HLMA_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'HLMA_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'HLMA_signal'] = -1
        return df
    
    # ------------------- CROSS_LINE: 两线交叉 -------------------
    
    def _signal_APZ(self, df):
        p = self.params
        upper, lower = gf.APZ(df['close'].values, df['high'].values, df['low'].values,
                              p['N'], p['M'], p['PARAM'])
        s_upper = pd.Series(upper, index=df.index)
        s_lower = pd.Series(lower, index=df.index)
        df['APZ_signal'] = 0
        df.loc[df['close'] > s_upper, 'APZ_signal'] = 1
        df.loc[df['close'] < s_lower, 'APZ_signal'] = -1
        return df
    
    def _signal_KC(self, df):
        p = self.params
        upper, lower = gf.KC(df['close'].values, df['high'].values, df['low'].values,
                             p['N'], p['M'])
        s_upper = pd.Series(upper, index=df.index)
        s_lower = pd.Series(lower, index=df.index)
        df['KC_signal'] = 0
        df.loc[df['close'] > s_upper, 'KC_signal'] = 1
        df.loc[df['close'] < s_lower, 'KC_signal'] = -1
        return df
    
    def _signal_ARRON(self, df):
        n = self.params['N']
        up, down, osc = gf.Arron(df['close'].values, df['high'].values, df['low'].values, n)
        s_up, s_down = pd.Series(up, index=df.index), pd.Series(down, index=df.index)
        df['ARRON_signal'] = 0
        df.loc[self._cross_over(s_up, s_down), 'ARRON_signal'] = 1
        df.loc[self._cross_under(s_up, s_down), 'ARRON_signal'] = -1
        return df
    
    # ------------------- 成交量指标 -------------------
    
    def _signal_MAAMT(self, df):
        n = self.params['N']
        val = gf.MAAMT(df['volume'].values, n)
        s = pd.Series(val, index=df.index)
        df['MAAMT_signal'] = 0
        df.loc[self._cross_over(df['volume'], s), 'MAAMT_signal'] = 1
        df.loc[self._cross_under(df['volume'], s), 'MAAMT_signal'] = -1
        return df
    
    def _signal_SROCVOL(self, df):
        p = self.params
        val = gf.SROCVOL(df['volume'].values, p['N'], p['M'])
        s = pd.Series(val, index=df.index)
        df['SROCVOL_signal'] = 0
        df.loc[self._cross_up(s, 0), 'SROCVOL_signal'] = 1
        df.loc[self._cross_down(s, 0), 'SROCVOL_signal'] = -1
        return df
    
    def _signal_BIASVOL(self, df):
        n = self.params['N']
        val = gf.BIASVOL(df['volume'].values, n)
        s = pd.Series(val, index=df.index)
        df['BIASVOL_signal'] = 0
        df.loc[self._cross_up(s, 0), 'BIASVOL_signal'] = 1
        df.loc[self._cross_down(s, 0), 'BIASVOL_signal'] = -1
        return df
    
    def _signal_MACDVOL(self, df):
        p = self.params
        macd_val, macd_signal = gf.MACDVOL(df['volume'].values, p['N1'], p['N2'], p['N3'])
        s1, s2 = pd.Series(macd_val, index=df.index), pd.Series(macd_signal, index=df.index)
        df['MACDVOL_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'MACDVOL_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'MACDVOL_signal'] = -1
        return df
    
    def _signal_ROCVOL(self, df):
        n = self.params['N']
        val = gf.ROCVOL(df['volume'].values, n)
        s = pd.Series(val, index=df.index)
        df['ROCVOL_signal'] = 0
        df.loc[self._cross_up(s, 0), 'ROCVOL_signal'] = 1
        df.loc[self._cross_down(s, 0), 'ROCVOL_signal'] = -1
        return df
    
    def _signal_RSIV(self, df):
        p = self.params
        val = gf.RSIV(df['volume'].values, df['close'].values, p['N'])
        s = pd.Series(val, index=df.index)
        df['RSIV_signal'] = 0
        df.loc[self._cross_up(s, 40), 'RSIV_signal'] = 1
        df.loc[self._cross_down(s, 60), 'RSIV_signal'] = -1
        return df
    
    def _signal_AMV(self, df):
        p = self.params
        amv1, amv2 = gf.AMV(df['open'].values, df['close'].values, df['volume'].values,
                            p['N1'], p['N2'])
        s1, s2 = pd.Series(amv1, index=df.index), pd.Series(amv2, index=df.index)
        df['AMV_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'AMV_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'AMV_signal'] = -1
        return df
    
    def _signal_VRAMT(self, df):
        n = self.params['N']
        val = gf.VRAMT(df['amount'].values, df['close'].values, n)
        s = pd.Series(val, index=df.index)
        df['VRAMT_signal'] = 0
        df.loc[self._cross_up(s, 250), 'VRAMT_signal'] = 1
        df.loc[self._cross_down(s, 300), 'VRAMT_signal'] = -1
        return df
    
    def _signal_CMF_V(self, df):
        n = self.params.get('N', 20)
        val = gf.CMF(df['close'].values, df['high'].values, df['low'].values,
                     df['volume'].values, n)
        s = pd.Series(val, index=df.index)
        df['CMF_V_signal'] = 0
        df.loc[self._cross_up(s, 0), 'CMF_V_signal'] = 1
        df.loc[self._cross_down(s, 0), 'CMF_V_signal'] = -1
        return df
    
    def _signal_PVI(self, df):
        p = self.params
        val = gf.PVI(df['close'].values, df['volume'].values, p['N'])
        s = pd.Series(val, index=df.index)
        df['PVI_signal'] = 0
        df.loc[self._cross_up(s, 0), 'PVI_signal'] = 1
        df.loc[self._cross_down(s, 0), 'PVI_signal'] = -1
        return df
    
    # ======================================================================
    # FB, ASI, CMF, ATR, WAD — 这些函数的 gf_factors 返回格式特殊，单独处理
    # ======================================================================
    
    def _signal_FB(self, df):
        p = self.params
        upper, lower = gf.FB(df['close'].values, df['high'].values, df['low'].values,
                             p['N'], p['PARAM'])
        df['FB_signal'] = 0
        df.loc[df['close'] > upper, 'FB_signal'] = 1
        df.loc[df['close'] < lower, 'FB_signal'] = -1
        return df
    
    def _signal_ASI(self, df):
        p = self.params
        # ASI 需要一个外部的 ASIMA 作为 trigger 线
        n_short = max(p['M'] // 2, 1)
        asi_val = gf.ASI(df['close'].values, df['high'].values, df['low'].values,
                         df['open'].values, p['N'], p['M'])
        # gf.ASI returns ASI only for the original call; SI is the combined version
        # 改用 SI 函数（已含 ASIMA）
        asi, asima = gf.SI(df['close'].values, df['high'].values, df['low'].values,
                           df['open'].values, p['N'], p['M'])
        s1, s2 = pd.Series(asi, index=df.index), pd.Series(asima, index=df.index)
        df['ASI_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'ASI_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'ASI_signal'] = -1
        return df
    
    def _signal_CMF(self, df):
        n = self.params['N']
        val = gf.CMF(df['close'].values, df['high'].values, df['low'].values,
                     df['volume'].values, n)
        s = pd.Series(val, index=df.index)
        df['CMF_signal'] = 0
        df.loc[self._cross_up(s, 0), 'CMF_signal'] = 1
        df.loc[self._cross_down(s, 0), 'CMF_signal'] = -1
        return df
    
    def _signal_ATR(self, df):
        n = self.params['N']
        val = gf.ATR(df['high'].values, df['low'].values, df['close'].values, n)
        s = pd.Series(val, index=df.index)
        # ATR 本身是波动率指标，用 ATR 与其均线的交叉来产生信号
        sma = gf.MA(val, n)
        s_ma = pd.Series(sma, index=df.index)
        df['ATR_signal'] = 0
        df.loc[self._cross_over(s, s_ma), 'ATR_signal'] = 1
        df.loc[self._cross_under(s, s_ma), 'ATR_signal'] = -1
        return df
    
    def _signal_WAD(self, df):
        p = self.params
        wad, wad_ma = gf.WAD(df['close'].values, df['high'].values, df['low'].values,
                              df['volume'].values, p['N'])
        s1, s2 = pd.Series(wad, index=df.index), pd.Series(wad_ma, index=df.index)
        df['WAD_signal'] = 0
        df.loc[self._cross_over(s1, s2), 'WAD_signal'] = 1
        df.loc[self._cross_under(s1, s2), 'WAD_signal'] = -1
        return df
