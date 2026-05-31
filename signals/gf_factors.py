import numpy as np
import pandas as pd


# ------------------ 0级：核心工具函数 --------------------------------------------
def RD(N, D=3):   return np.round(N, D)  # 四舍五入取3位小数


def RET(S, N=1):  return np.array(S)[-N]  # 返回序列倒数第N个值,默认返回最后一个


def ABS(S):      return np.abs(S)  # 返回N的绝对值


def LN(S):       return np.log(S)  # 求底是e的自然对数,


def POW(S, N):    return np.power(S, N)  # 求S的N次方


def SQRT(S):     return np.sqrt(S)  # 求S的平方根


def SIN(S):      return np.sin(S)  # 求S的正弦值（弧度)


def COS(S):      return np.cos(S)  # 求S的余弦值（弧度)


def TAN(S):      return np.tan(S)  # 求S的正切值（弧度)


def MAX(S1, S2):  return np.maximum(S1, S2)  # 序列max


def MIN(S1, S2):  return np.minimum(S1, S2)  # 序列min


def IF(S, A, B):   return np.where(S, A, B)  # 序列布尔判断 return=A  if S==True  else  B


def REF(S, N=1):  # 对序列整体下移动N,返回序列(shift后会产生NAN)
    return pd.Series(S).shift(N).values


def DIFF(S, N=1):  # 前一个值减后一个值,前面会产生nan
    return pd.Series(S).diff(N).values  # np.diff(S)直接删除nan，会少一行


def STD(S, N):  # 求序列的N日标准差，返回序列
    return pd.Series(S).rolling(N).std(ddof=0).values


def SUM(S, N):  # 对序列求N天累计和，返回序列    N=0对序列所有依次求和
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values

def CUM_SUM(S):  # 对序列求累计和，返回序列
    return SUM(S, 0)

def CONST(S):  # 返回序列S最后的值组成常量序列
    return np.full(len(S), S[-1])


def HHV(S, N):  # HHV(C, 5) 最近5天收盘最高价
    return pd.Series(S).rolling(N).max().values


def LLV(S, N):  # LLV(C, 5) 最近5天收盘最低价
    return pd.Series(S).rolling(N).min().values


def HHVBARS(S, N):  # 求N周期内S最高值到当前周期数, 返回序列
    return pd.Series(S).rolling(N).apply(lambda x: np.argmax(x[::-1]), raw=True).values


def LLVBARS(S, N):  # 求N周期内S最低值到当前周期数, 返回序列
    return pd.Series(S).rolling(N).apply(lambda x: np.argmin(x[::-1]), raw=True).values


def MA(S, N):  # 求序列的N日简单移动平均值，返回序列
    return pd.Series(S).rolling(N).mean().values


def EMA(S, N):  # 指数移动平均,为了精度 S>4*N  EMA至少需要120周期     alpha=2/(span+1)
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def SMA(S, N, M=1):  # 中国式的SMA,至少需要120周期才精确 (雪球180周期)    alpha=1/(1+com)
    return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values  # com=N-M/M


def WMA(S, N):  # 通达信S序列的N日加权移动平均 Yn = (1*X1+2*X2+3*X3+...+n*Xn)/(1+2+3+...+Xn)
    return pd.Series(S).rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1), raw=True).values


def DMA(S, A):  # 求S的动态移动平均，A作平滑因子,必须 0<A<1  (此为核心函数，非指标）
    if isinstance(A, (int, float)):  return pd.Series(S).ewm(alpha=A, adjust=False).mean().values
    A = np.array(A);
    A[np.isnan(A)] = 1.0;
    Y = np.zeros(len(S));
    Y[0] = S[0]
    for i in range(1, len(S)): Y[i] = A[i] * S[i] + (1 - A[i]) * Y[i - 1]  # A支持序列 by jqz1226
    return Y


def AVEDEV(S, N):  # 平均绝对偏差  (序列与其平均值的绝对差的平均值)
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values


def SLOPE(S, N):  # 返S序列N周期回线性回归斜率
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N), x, deg=1)[0], raw=True).values


def FORCAST(S, N):  # 返回S序列N周期回线性回归后的预测值， jqz1226改进成序列出
    return pd.Series(S).rolling(N).apply(lambda x: np.polyval(np.polyfit(range(N), x, deg=1), N - 1), raw=True).values


def LAST(S, A, B):  # 从前A日到前B日一直满足S_BOOL条件, 要求A>B & A>0 & B>=0
    return np.array(pd.Series(S).rolling(A + 1).apply(lambda x: np.all(x[::-1][B:]), raw=True), dtype=bool)


# ------------------   1级：应用层函数(通过0级核心函数实现）使用方法请参考通达信--------------------------------
def COUNT(S, N):  # COUNT(CLOSE>O, N):  最近N天满足S_BOO的天数  True的天数
    return SUM(S, N)


def EVERY(S, N):  # EVERY(CLOSE>O, 5)   最近N天是否都是True
    return IF(SUM(S, N) == N, True, False)


def EXIST(S, N):  # EXIST(CLOSE>3010, N=5)  n日内是否存在一天大于3000点
    return IF(SUM(S, N) > 0, True, False)


def FILTER(S, N):  # FILTER函数，S满足条件后，将其后N周期内的数据置为0, FILTER(C==H,5)
    for i in range(len(S)): S[i + 1:i + 1 + N] = 0 if S[i] else S[i + 1:i + 1 + N]
    return S  # 例：FILTER(C==H,5) 涨停后，后5天不再发出信号


def BARSLAST(S):  # 上一次条件成立到当前的周期, BARSLAST(C/REF(C,1)>=1.1) 上一次涨停到今天的天数
    M = np.concatenate(([0], np.where(S, 1, 0)))
    for i in range(1, len(M)):  M[i] = 0 if M[i] else M[i - 1] + 1
    return M[1:]


def BARSLASTCOUNT(S):  # 统计连续满足S条件的周期数        by jqz1226
    rt = np.zeros(len(S) + 1)  # BARSLASTCOUNT(CLOSE>OPEN)表示统计连续收阳的周期数
    for i in range(len(S)): rt[i + 1] = rt[i] + 1 if S[i] else rt[i + 1]
    return rt[1:]


def BARSSINCEN(S, N):  # N周期内第一次S条件成立到现在的周期数,N为常量  by jqz1226
    return pd.Series(S).rolling(N).apply(lambda x: N - 1 - np.argmax(x) if np.argmax(x) or x[0] else 0,
                                         raw=True).fillna(0).values.astype(int)


def CROSS(S1, S2):  # 判断向上金叉穿越 CROSS(MA(C,5),MA(C,10))  判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # 不使用0级函数,移植方便  by jqz1226


def LONGCROSS(S1, S2, N):  # 两条线维持一定周期后交叉,S1在N周期内都小于S2,本周期从S1下方向上穿过S2时返回1,否则返回0
    return np.array(np.logical_and(LAST(S1 < S2, N, 1), (S1 > S2)), dtype=bool)  # N=1时等同于CROSS(S1, S2)


def VALUEWHEN(S, X):  # 当S条件成立时,取X的当前值,否则取VALUEWHEN的上个成立时的X值   by jqz1226
    return pd.Series(np.where(S, X, np.nan)).ffill().values


def BETWEEN(S, A, B):  # S处于A和B之间时为真。 包括 A<S<B 或 A>S>B
    return ((A < S) & (S < B)) | ((A > S) & (S > B))


def TOPRANGE(S):  # TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值 by jqz1226
    rt = np.zeros(len(S))
    for i in range(1, len(S)):  rt[i] = np.argmin(np.flipud(S[:i] < S[i]))
    return rt.astype('int')


def LOWRANGE(S):  # LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值 by jqz1226
    rt = np.zeros(len(S))
    for i in range(1, len(S)):  rt[i] = np.argmin(np.flipud(S[:i] > S[i]))
    return rt.astype('int')


# ------------------   2级：技术指标函数(全部通过0级，1级函数实现） ------------------------------
def DMI(CLOSE,HIGH,LOW,M1=14,M2=6):               #动向指标：结果和同花顺，通达信完全一致
    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
    HD = HIGH - REF(HIGH, 1);     LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / TR;         MDI = DMM * 100 / TR
    ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    return PDI, MDI, ADX, ADXR

def AD(CLOSE, HIGH, LOW, VOL):
    """
    计算累积/派发线（AD）指标，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param VOL: 成交量序列
    :return: AD指标序列（numpy.ndarray）
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐和数据一致性
    for s in [CLOSE, HIGH, LOW, VOL]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            VOL = pd.Series(VOL)
            break
    
    # 2. 计算CMF的分子和分母
    cmf_numerator = 2 * CLOSE - HIGH - LOW  # 分子：2*收盘价 - 最高价 - 最低价
    cmf_denominator = HIGH - LOW            # 分母：最高价 - 最低价
    
    # 3. 处理除零问题：分母为0或极接近0时，CMF设为0（价格无波动时资金流向视为中性）
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    valid = (cmf_denominator > epsilon) & ~np.isnan(cmf_denominator) & ~np.isnan(cmf_numerator)
    CMF = np.where(valid, cmf_numerator / cmf_denominator, 0.0)  # 无效时设为0
    
    # 4. 计算AD指标（累积和）
    AD = np.zeros(len(CLOSE))
    # 处理第一个值（避免索引越界）
    AD[0] = CMF[0] * VOL.iloc[0] if len(CLOSE) > 0 else 0.0
    # 累积计算后续值
    for i in range(1, len(CLOSE)):
        AD[i] = AD[i-1] + CMF[i] * VOL.iloc[i]
    
    return AD

# 表34:价格动量指标

def DPO(CLOSE, N=20):  # 区间震荡线
    """DPO=CLOSE-REF(MA(CLOSE,N),N/2+1)"""
    return CLOSE - REF(MA(CLOSE, N), N // 2 + 1)


def ER(CLOSE, HIGH, LOW, N=20):  # Elder Ray Index
    """BullPower=HIGH-EMA(CLOSE,N)
       BearPower=LOW-EMA(CLOSE,N)"""
    BullPower = HIGH - EMA(CLOSE, N)
    BearPower = LOW - EMA(CLOSE, N)
    return BullPower, BearPower


def TII(CLOSE, N1=40, M=20, N2=9):  # Trend Intensity Index
    """CLOSE_MA=MA(CLOSE,N1)
       DEV=CLOSE-CLOSE_MA
       DEVPOS=IF(DEV>0,DEV,0)
       DEVNEG=IF(DEV<0,-DEV,0)
       SUMPOS=SUM(DEVPOS,M)
       SUMNEG=SUM(DEVNEG,M)
       TII=100*SUMPOS/(SUMPOS+SUMNEG)"""
    CLOSE_MA = MA(CLOSE, N1)
    DEV = CLOSE - CLOSE_MA
    DEVPOS = MAX(DEV, 0)
    DEVNEG = MAX(-DEV, 0)
    SUMPOS = SUM(DEVPOS, M)
    SUMNEG = SUM(DEVNEG, M)
    return 100 * SUMPOS / (SUMPOS + SUMNEG)


def PO(CLOSE, N1=9, N2=26):  # Price Oscillator
    """PO=(EMA(CLOSE,N1)-EMA(CLOSE,N2))/EMA(CLOSE,N2)*100"""
    EMA_Short = EMA(CLOSE, N1)
    EMA_Long = EMA(CLOSE, N2)
    return (EMA_Short - EMA_Long) / EMA_Long * 100


def MADisplaced(CLOSE, N=20, M=10):  # Displaced Moving Average
    """MADisplaced=REF(MA(CLOSE,N),M)"""
    return REF(MA(CLOSE, N), M)


def T3(CLOSE, N=20, VA=0.7):  # T3指标
    """T3=EMA(T2,N)*(1+VA)-EMA(EMA(T2,N),N)*VA
       T2=EMA(T1,N)*(1+VA)-EMA(EMA(T1,N),N)*VA
       T1=EMA(CLOSE,N)"""
    T1 = EMA(CLOSE, N)
    T2 = EMA(T1, N) * (1 + VA) - EMA(EMA(T1, N), N) * VA
    return EMA(T2, N) * (1 + VA) - EMA(EMA(T2, N), N) * VA


def POS(CLOSE, N=100):  # 位置指标
    """PRICE=(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)
       POS=(PRICE-MIN(PRICE,N))/(MAX(PRICE,N)-MIN(PRICE,N))"""
    PRICE = (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N)
    return (PRICE - MIN(PRICE, N)) / (MAX(PRICE, N) - MIN(PRICE, N))


def PAC(HIGH, LOW, N1=20, N2=20):  # PAC指标
    """UPPER=SMA(HIGH,N1,1)
       LOWER=SMA(LOW,N2,1)"""
    UPPER = SMA(HIGH, N1, 1)
    LOWER = SMA(LOW, N2, 1)
    return UPPER, LOWER


def ADTM(OPEN, HIGH, LOW, N=20):  # ADTM指标
    """DTM=IF(OPEN>REF(OPEN,1),MAX(HIGH-OPEN,OPEN-REF(OPEN,1)),0)
       DBM=IF(OPEN<REF(OPEN,1),MAX(OPEN-LOW,REF(OPEN,1)-OPEN),0)
       ADTM=(SUM(DTM,N)-SUM(DBM,N))/MAX(SUM(DTM,N),SUM(DBM,N))"""
    DTM = IF(OPEN > REF(OPEN, 1), MAX(HIGH - OPEN, OPEN - REF(OPEN, 1)), 0)
    DBM = IF(OPEN < REF(OPEN, 1), MAX(OPEN - LOW, REF(OPEN, 1) - OPEN), 0)
    STM = SUM(DTM, N)
    SBM = SUM(DBM, N)
    return (STM - SBM) / MAX(STM, SBM)


def ZLMACD(CLOSE, N1=20, N2=100):  # ZLMACD指标
    """ZLMACD=(2*EMA(CLOSE,N1)-EMA(EMA(CLOSE,N1),N1))-
              (2*EMA(CLOSE,N2)-EMA(EMA(CLOSE,N2),N2))"""
    EMA1 = 2 * EMA(CLOSE, N1) - EMA(EMA(CLOSE, N1), N1)
    EMA2 = 2 * EMA(CLOSE, N2) - EMA(EMA(CLOSE, N2), N2)
    return EMA1 - EMA2


def TMA(CLOSE, N=20):  # 三重移动平均
    """TMA=MA(MA(MA(CLOSE,N),N),N)"""
    return MA(MA(MA(CLOSE, N), N), N)


def TYP(CLOSE, HIGH, LOW, N1=10, N2=30):  # 典型价格移动平均
    """TYP=(CLOSE+HIGH+LOW)/3
       TYPMA1=EMA(TYP,N1)
       TYPMA2=EMA(TYP,N2)"""
    TYP = (CLOSE + HIGH + LOW) / 3
    return EMA(TYP, N1), EMA(TYP, N2)


def KDJD(CLOSE, HIGH, LOW, N=9, M=3):
    """
    计算KDJD指标，修复除零警告
    CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    HIGH: 最高价序列
    LOW: 最低价序列
    N: 初始窗口大小（默认9）
    M: 二次平滑窗口大小（默认3）
    返回：K值序列、D值序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    for s in [CLOSE, HIGH, LOW]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break  # 统一转换后退出检查
    
    # 2. 实现正确的LLV（滚动最小值）和HHV（滚动最大值）
    def LLV(X, window):
        """滚动window期内的最小值，数据不足时为NaN"""
        return X.rolling(window=window, min_periods=window).min()
    
    def HHV(X, window):
        """滚动window期内的最大值，数据不足时为NaN"""
        return X.rolling(window=window, min_periods=window).max()
    
    # 3. 计算基础指标
    LOW_N = LLV(LOW, N)  # N期最低价
    HIGH_N = HHV(HIGH, N)  # N期最高价
    
    # 处理HIGH_N - LOW_N为0的情况（避免Stochastics除零）
    price_range = HIGH_N - LOW_N
    epsilon_price = 1e-10
    valid_price_range = (price_range > epsilon_price) & ~np.isnan(price_range)
    Stochastics = np.where(
        valid_price_range,
        (CLOSE - LOW_N) / price_range * 100,
        np.nan
    )
    Stochastics = pd.Series(Stochastics, index=CLOSE.index)  # 保持索引
    
    # 4. 计算Stochastics_DOUBLE，修复除零警告
    # M期内Stochastics的最高和最低
    stoch_high = HHV(Stochastics, M)
    stoch_low = LLV(Stochastics, M)
    denominator = stoch_high - stoch_low  # 分母
    
    # 强化有效性检查：分母非NaN+绝对值>epsilon+分子非NaN
    epsilon = 1e-10
    valid_denominator = (
        ~np.isnan(denominator) 
        & (np.abs(denominator) > epsilon)
        & ~np.isnan(Stochastics)
        & ~np.isnan(stoch_low)
    )
    
    # 使用np.divide的where参数，只在有效时执行除法，彻底避免无效运算
    Stochastics_DOUBLE = np.divide(
        (Stochastics - stoch_low) * 100,  # 分子
        denominator,                      # 分母
        out=np.full_like(Stochastics, np.nan),  # 无效时输出NaN
        where=valid_denominator               # 仅在有效条件下执行除法
    )
    
    # 5. 计算K和D（SMA为简单移动平均）
    def SMA(X, window, min_periods=1):
        """简单移动平均"""
        return pd.Series(X).rolling(window=window, min_periods=min_periods).mean()
    
    K = SMA(Stochastics_DOUBLE, 3, 1)
    D = SMA(K, 3, 1)
    
    return K, D


def VMA(CLOSE, HIGH, LOW, OPEN, N=20):  # 成交量加权移动平均
    """PRICE=(HIGH+LOW+OPEN+CLOSE)/4
       VMA=MA(PRICE,N)"""
    PRICE = (HIGH + LOW + OPEN + CLOSE) / 4
    return MA(PRICE, N)


def BIAS(CLOSE, N=6):  # 乖离率
    """BIAS=(CLOSE-MA(CLOSE,N))/MA(CLOSE,N)*100"""
    return (CLOSE - MA(CLOSE, N)) / MA(CLOSE, N) * 100


def WMA(CLOSE, N=20):  # 加权移动平均
    """WMA=(N*CLOSE+(N-1)*REF(CLOSE,1)+...+1*REF(CLOSE,N-1))/
          (N*(N+1)/2)"""
    # 生成从 N 到 1 的权重
    weights = np.arange(N, 0, -1)
    weighted_close = np.zeros_like(CLOSE)
    for i in range(N):
        if i == 0:
            weighted_close += weights[i] * CLOSE
        else:
            weighted_close += weights[i] * REF(CLOSE, i)

    # 计算加权和的 N 天累计和
    weighted_sum = SUM(weighted_close, N)
    # 计算权重总和
    total_weight = SUM(weights, N=0)[-1]  # 因为权重是固定序列，N=0 求所有元素和
    return weighted_sum / total_weight


def DDI(CLOSE, HIGH, LOW, N=40):  # DDI指标
    """HL=HIGH+LOW
       HIGH_ABS=ABS(HIGH-REF(HIGH,1))
       LOW_ABS=ABS(LOW-REF(LOW,1))
       DMZ=IF(HL>REF(HL,1),MAX(HIGH_ABS,LOW_ABS),0)
       DMF=IF(HL<REF(HL,1),MAX(HIGH_ABS,LOW_ABS),0)
       DIZ=SUM(DMZ,N)/(SUM(DMZ,N)+SUM(DMF,N))
       DIF=SUM(DMF,N)/(SUM(DMZ,N)+SUM(DMF,N))
       DDI=DIZ-DIF"""
    HL = HIGH + LOW
    HIGH_ABS = ABS(HIGH - REF(HIGH, 1))
    LOW_ABS = ABS(LOW - REF(LOW, 1))
    DMZ = IF(HL > REF(HL, 1), MAX(HIGH_ABS, LOW_ABS), 0)
    DMF = IF(HL < REF(HL, 1), MAX(HIGH_ABS, LOW_ABS), 0)
    DIZ = SUM(DMZ, N) / (SUM(DMZ, N) + SUM(DMF, N))
    DIF = SUM(DMF, N) / (SUM(DMZ, N) + SUM(DMF, N))
    return DIZ - DIF


def HMA(HIGH, N=20):  # 最高价移动平均
    """HMA=MA(HIGH,N)"""
    return MA(HIGH, N)


def SROC(CLOSE, N=13, M=21):  # 平滑ROC
    """EMAP=EMA(CLOSE,N)
       SROC=(EMAP-REF(EMAP,M))/REF(EMAP,M)"""
    EMAP = EMA(CLOSE, N)
    return (EMAP - REF(EMAP, M)) / REF(EMAP, M)


def EXPMA(CLOSE, N=12):  # 指数移动平均
    """EXPMA=EMA(CLOSE,N)"""
    return EMA(CLOSE, N)


def DC(HIGH, LOW, N=20):  # DC指标
    """UPPER=MAX(HIGH,N)
       LOWER=MIN(LOW,N)
       MIDDLE=(UPPER+LOWER)/2"""
    UPPER = HHV(HIGH, N)
    LOWER = LLV(LOW, N)
    return UPPER, LOWER, (UPPER + LOWER) / 2

def VIDYA(CLOSE, N=10):  # 修复版VIDYA指标（变异性指数动态平均线）
    """
    计算VIDYA指标，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N: 计算周期（默认10）
    :return: VIDYA指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的REF（前N期移位）和滚动SUM（N期累积和）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)，前n期为NaN"""
        return X.shift(n)
    
    def rolling_sum(X, window):
        """N期滚动求和，仅窗口内有window个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).sum()
    
    # 3. 计算VI的分子和分母
    # 分子：ABS(CLOSE - 前N期收盘价)
    ref_close_n = REF(CLOSE, N)  # 前N期收盘价
    vi_numerator = np.abs(CLOSE - ref_close_n)  # 绝对值用np.abs确保正确
    
    # 分母：N期内每日收盘价波动绝对差的滚动和
    daily_change = np.abs(CLOSE - REF(CLOSE, 1))  # 每日收盘价与前一日的绝对差
    vi_denominator = rolling_sum(daily_change, N)  # N期滚动求和（分母）
    
    # 4. 安全计算VI：过滤分母为0或NaN的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    # 有效性条件：分母非0+非NaN + 分子非NaN（避免无效运算）
    valid = (
        (vi_denominator > epsilon) & ~np.isnan(vi_denominator) &
        ~np.isnan(vi_numerator)
    )
    
    # 仅在有效时计算VI，无效时设为NaN（避免除零）
    VI = np.where(valid, vi_numerator / vi_denominator, np.nan)
    VI = pd.Series(VI, index=CLOSE.index)  # 保持索引一致
    
    # 5. 计算最终VIDYA指标
    ref_close_1 = REF(CLOSE, 1)  # 前1期收盘价
    VIDYA = VI * CLOSE + (1 - VI) * ref_close_1
    
    return VIDYA


def Qstick(CLOSE, OPEN, N=20):  # Qstick指标
    """Qstick=MA(CLOSE-OPEN,N)"""
    return MA(CLOSE - OPEN, N)


def FB(CLOSE, HIGH, LOW, N=20, PARAM=1.618):  # Fibonacci Bands
    """TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
       ATR=MA(TR,N)
       MIDDLE=MA(CLOSE,N)
       UPPER1=MIDDLE+1.618*ATR
       UPPER2=MIDDLE+2.618*ATR
       UPPER3=MIDDLE+4.236*ATR
       LOWER1=MIDDLE-1.618*ATR
       LOWER2=MIDDLE-2.618*ATR
       LOWER3=MIDDLE-4.236*ATR"""
    temp_max = MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1)))
    TR = MAX(temp_max, ABS(LOW - REF(CLOSE, 1)))
    ATR = MA(TR, N)
    MIDDLE = MA(CLOSE, N)
    return (MIDDLE + PARAM * ATR, MIDDLE - PARAM * ATR)


def DEMA(CLOSE, N=60):  # 双重指数移动平均
    """DEMA=2*EMA(CLOSE,N)-EMA(EMA(CLOSE,N),N)"""
    EMA1 = EMA(CLOSE, N)
    return 2 * EMA1 - EMA(EMA1, N)


def APZ(CLOSE, HIGH, LOW, N=10, M=20, PARAM=2):  # APZ指标
    """VOL=EMA(EMA(HIGH-LOW,N),N)
       UPPER=EMA(EMA(CLOSE,M),M)+PARAM*VOL
       LOWER=EMA(EMA(CLOSE,M),M)-PARAM*VOL"""
    VOL = EMA(EMA(HIGH - LOW, N), N)
    EMAClose = EMA(EMA(CLOSE, M), M)
    return EMAClose + PARAM * VOL, EMAClose - PARAM * VOL


def ASI(CLOSE, HIGH, LOW, OPEN, N=20, M=20):  # ASI指标（修复版）
    """
    计算振动升降指数（ASI）及平均线（ASIMA），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param OPEN: 开盘价序列
    :param N: 价格波动下限参数（默认20）
    :param M: ASIMA的移动平均窗口（默认20）
    :return: ASI序列、ASIMA序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    for s in [CLOSE, HIGH, LOW, OPEN]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            OPEN = pd.Series(OPEN)
            break
    
    # 2. 实现基础函数（替代自定义的REF、ABS、MAX、IF等）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    # 3. 计算A、B、C、D
    ref_close_1 = REF(CLOSE, 1)  # 前1期收盘价
    ref_low_1 = REF(LOW, 1)      # 前1期最低价
    ref_open_1 = REF(OPEN, 1)    # 前1期开盘价
    
    A = np.abs(HIGH - ref_close_1)  # A = ABS(HIGH - REF(CLOSE,1))
    B = np.abs(LOW - ref_close_1)   # B = ABS(LOW - REF(CLOSE,1))
    C = np.abs(HIGH - ref_low_1)    # C = ABS(HIGH - REF(LOW,1))
    D = np.abs(ref_close_1 - ref_open_1)  # D = ABS(REF(CLOSE,1)-REF(OPEN,1))
    
    # 4. 计算K、R1、R2、R3、R
    K = np.maximum(A, B)  # K = MAX(A,B)
    R1 = A + 0.5 * B + 0.25 * D
    R2 = B + 0.5 * A + 0.25 * D
    R3 = C + 0.25 * D
    
    # R的条件判断（替代IF函数）
    # 条件1：C >= A 且 C >= B → R=R3
    # 条件2：不满足条件1，但A >= B → R=R1
    # 条件3：其他情况 → R=R2
    cond1 = (C >= A) & (C >= B)
    cond2 = ~cond1 & (A >= B)
    R = np.where(cond1, R3, np.where(cond2, R1, R2))
    
    # 5. 计算第二个分母：MAX(HIGH-LOW, N)
    price_range = HIGH - LOW
    denominator2 = np.maximum(price_range, N)  # 确保至少为N（避免0）
    
    # 6. 计算SI的分子
    si_numerator = 50 * (
        (CLOSE - ref_close_1) + 
        (ref_close_1 - ref_open_1) + 
        0.5 * (CLOSE - OPEN)
    )
    
    # 7. 处理除法无效值：过滤R=0或NaN、denominator2=0或NaN的情况
    epsilon = 1e-10
    valid = (
        (R > epsilon) & ~np.isnan(R) &  # R有效（非0且非NaN）
        (denominator2 > epsilon) & ~np.isnan(denominator2) &  # denominator2有效
        ~np.isnan(si_numerator) & ~np.isnan(K)  # 分子和K有效
    )
    
    # 安全计算SI：仅在有效时执行除法，否则为NaN
    SI = np.where(
        valid,
        si_numerator / R * K / denominator2,
        np.nan
    )
    
    # 8. 计算ASI（累积和）和ASIMA（移动平均）
    ASI = pd.Series(SI).cumsum()  # 替代CUM_SUM
    
    def MA(X, window):
        """简单移动平均"""
        return pd.Series(X).rolling(window=window, min_periods=1).mean()
    
    ASIMA = MA(ASI, M)
    
    return ASI, ASIMA


def Arron(CLOSE, HIGH, LOW, N=20):  # Arron指标
    """HIGH_LEN=BARSLAST(HIGH==HHV(HIGH,N))
       LOW_LEN=BARSLAST(LOW==LLV(LOW,N))
       ArronUp=(N-HIGH_LEN)/N*100
       ArronDown=(N-LOW_LEN)/N*100
       ArronOs=ArronUp-ArronDown"""
    HIGH_LEN = BARSLAST(HIGH == HHV(HIGH, N))
    LOW_LEN = BARSLAST(LOW == LLV(LOW, N))
    ArronUp = (N - HIGH_LEN) / N * 100
    ArronDown = (N - LOW_LEN) / N * 100
    return ArronUp, ArronDown, ArronUp - ArronDown


def KC(CLOSE, HIGH, LOW, N=14, M=20):  # Keltner Channel
    """TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-LOW))
       ATR=MA(TR,N)
       Middle=EMA(CLOSE,M)
       Upper=Middle+2*ATR
       Lower=Middle-2*ATR"""
    temp_max = MAX(ABS(HIGH - LOW), ABS(HIGH - REF(CLOSE, 1)))
    TR = MAX(temp_max, ABS(REF(CLOSE, 1) - LOW))
    ATR = MA(TR, N)
    Middle = EMA(CLOSE, M)
    return Middle + 2 * ATR, Middle - 2 * ATR


def MTM(CLOSE, N=60):  # 动量指标
    """MTM=CLOSE-REF(CLOSE,N)"""
    return CLOSE - REF(CLOSE, N)


def CR(CLOSE, HIGH, LOW, N=20):  # CR指标
    """TYP=(HIGH+LOW+CLOSE)/3
       H=MAX(HIGH-REF(TYP,1),0)
       L=MAX(REF(TYP,1)-LOW,0)
       CR=SUM(H,N)/SUM(L,N)*100"""
    TYP = (HIGH + LOW + CLOSE) / 3
    H = MAX(HIGH - REF(TYP, 1), 0)
    L = MAX(REF(TYP, 1) - LOW, 0)
    return SUM(H, N) / SUM(L, N) * 100


def BOP(CLOSE, OPEN, HIGH, LOW, N=20):  # 平衡交易量指标（BOP）
    """
    计算平衡交易量指标（BOP），修复除零警告
    CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    OPEN: 开盘价序列
    HIGH: 最高价序列
    LOW: 最低价序列
    N: 移动平均窗口（默认20）
    返回：BOP序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    for s in [CLOSE, OPEN, HIGH, LOW]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            OPEN = pd.Series(OPEN)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break  # 统一转换后退出检查
    
    # 2. 计算分子和分母
    numerator = CLOSE - OPEN  # 分子：收盘价 - 开盘价
    denominator = HIGH - LOW  # 分母：最高价 - 最低价
    
    # 3. 定义有效条件：分母非0且非NaN（避免除零和无效值）
    epsilon = 1e-10  # 极小值，处理浮点数精度问题（如0.0000000001视为0）
    valid = (denominator > epsilon) & ~np.isnan(denominator) & ~np.isnan(numerator)
    
    # 4. 安全计算比率：仅在有效条件下执行除法，否则返回NaN
    ratio = np.where(valid, numerator / denominator, np.nan)
    
    # 5. 计算移动平均（MA）：使用简单移动平均，忽略NaN
    def MA(X, window):
        """简单移动平均，窗口内数据不足时返回NaN"""
        return pd.Series(X).rolling(window=window, min_periods=window).mean()
    
    BOP = MA(ratio, N)
    return BOP


def HULLMA(CLOSE, N=20):  # HULL移动平均
    """X=2*EMA(CLOSE,N//2)-EMA(CLOSE,N)
       HULLMA=EMA(X,INT(SQRT(N)))"""
    X = 2 * EMA(CLOSE, N // 2) - EMA(CLOSE, N)
    return EMA(X, int(np.sqrt(N)))


def COPP(CLOSE, N1=10, N2=20, M=5):  # COPP指标
    """RC=100*((CLOSE-REF(CLOSE,N1))/REF(CLOSE,N1)+(CLOSE-REF(CLOSE,N2))/REF(CLOSE,N2))
       COPP=WMA(RC,M)"""
    RC = 100 * ((CLOSE - REF(CLOSE, N1)) / REF(CLOSE, N1) + (CLOSE - REF(CLOSE, N2)) / REF(CLOSE, N2))
    return WMA(RC, M)


def ENV(CLOSE, N=20, PARAM=0.1):  # 包络线
    """MAC=MA(CLOSE,N)
       UPPER=MAC*(1+PARAM)
       LOWER=MAC*(1-PARAM)"""
    MAC = MA(CLOSE, N)
    return MAC * (1 + PARAM), MAC * (1 - PARAM)


def RSIH(CLOSE, N1=40, N2=20):  # 修复版RSIH指标
    """
    计算RSIH指标，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N1: RSI的计算周期（默认40）
    :param N2: RSI信号线（EMA）的周期（默认20）
    :return: RSIH指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def SMA(X, window, min_periods=1):
        """简单移动平均（滚动窗口），自动忽略NaN"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    def EMA(X, window, min_periods=1):
        """指数移动平均（滚动窗口），自动忽略NaN"""
        return X.ewm(span=window, min_periods=min_periods, adjust=False).mean()
    
    # 3. 计算RSI的分子和分母
    # 每日价格变动：CLOSE - 前1期收盘价
    price_change = CLOSE - REF(CLOSE, 1)
    # 上涨幅度（下跌时取0）
    up_change = np.maximum(price_change, 0)  # 替代MAX(price_change, 0)
    # 价格变动绝对值
    abs_change = np.abs(price_change)        # 替代ABS(price_change)
    
    # 分子：N1期上涨幅度的移动平均
    sma_up = SMA(up_change, N1, min_periods=N1)
    # 分母：N1期价格变动绝对值的移动平均
    sma_abs = SMA(abs_change, N1, min_periods=N1)
    
    # 4. 安全计算RSI：过滤分母为0或NaN的情况
    epsilon = 1e-10  # 处理浮点数精度误差
    # 有效性条件：分母非0+非NaN + 分子非NaN
    valid = (
        (sma_abs > epsilon) & ~np.isnan(sma_abs) &
        ~np.isnan(sma_up)
    )
    
    # 仅在有效时计算RSI，无效时设为50（价格无波动时视为中性）
    RSI = np.where(
        valid,
        (sma_up / sma_abs) * 100,
        50.0  # 特殊处理：N1期无波动时，RSI理论上为50（中性）
    )
    RSI = pd.Series(RSI, index=CLOSE.index)
    
    # 5. 计算RSI信号线（EMA）和RSIH
    RSI_SIGNAL = EMA(RSI, N2, min_periods=N2)  # RSI的指数移动平均
    RSIH = RSI - RSI_SIGNAL  # RSI与信号线的差值
    
    return RSIH


def HLMA(HIGH, LOW, N1=20, N2=20):  # 高低点移动平均
    """HMA=MA(HIGH,N1)
       LMA=MA(LOW,N2)"""
    return MA(HIGH, N1), MA(LOW, N2)


def TSI(CLOSE, N1=25, N2=13):  # 修复版TSI指标（真实强弱指数）
    """
    计算TSI指标，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N1: 第一次EMA的周期（默认25）
    :param N2: 第二次EMA的周期（默认13）
    :return: TSI指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def EMA(X, window, min_periods=1):
        """指数移动平均（EMA）：使用pandas.ewm实现，adjust=False确保符合指标定义"""
        return X.ewm(span=window, min_periods=min_periods, adjust=False).mean()
    
    # 3. 计算价格变动及绝对值
    price_change = CLOSE - REF(CLOSE, 1)  # 每日收盘价变动（涨为正，跌为负）
    abs_change = np.abs(price_change)     # 每日收盘价变动的绝对值（总波动）
    
    # 4. 计算分子（二次EMA的净变动）和分母（二次EMA的总变动）
    # 分子：先N1期EMA，再N2期EMA
    ema1_numerator = EMA(price_change, N1, min_periods=N1)  # 第一次EMA
    numerator = EMA(ema1_numerator, N2, min_periods=N2)      # 第二次EMA
    
    # 分母：先N1期EMA，再N2期EMA
    ema1_denominator = EMA(abs_change, N1, min_periods=N1)   # 第一次EMA
    denominator = EMA(ema1_denominator, N2, min_periods=N2)   # 第二次EMA
    
    # 5. 安全计算TSI：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    # 有效性条件：分母非0+非NaN + 分子非NaN
    valid = (
        (denominator > epsilon) & ~np.isnan(denominator) &
        ~np.isnan(numerator)
    )
    
    # 特殊处理：当价格无波动时（分子≈0且分母≈0），TSI理论上为0（无涨跌）
    is_flat = (np.abs(numerator) < epsilon) & (np.abs(denominator) < epsilon)
    
    # 安全计算：有效时正常除法，价格无波动时设为0，其他情况设为NaN
    TSI = np.where(
        valid,
        (numerator / denominator) * 100,
        np.where(is_flat, 0.0, np.nan)
    )
    
    return pd.Series(TSI, index=CLOSE.index)


def BIAS36(CLOSE, N=6):  # 三六乖离
    """BIAS36=MA(CLOSE,3)-MA(CLOSE,6)"""
    return MA(CLOSE, 3) - MA(CLOSE, 6)


def UOS(CLOSE, HIGH, LOW, N1=7, N2=14, N3=28):  # 终极摆动指标（UOS）
    """
    计算终极摆动指标（UOS），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param N1: 短周期（默认7）
    :param N2: 中周期（默认14）
    :param N3: 长周期（默认28）
    :return: UOS指标序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    for s in [CLOSE, HIGH, LOW]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break
    
    # 2. 实现基础函数（替代自定义函数，确保逻辑正确）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def rolling_sum(X, window):
        """滚动window期求和，数据不足时返回NaN"""
        return pd.Series(X).rolling(window=window, min_periods=window).sum()
    
    # 3. 计算TH、TL、TR、XR
    ref_close_1 = REF(CLOSE, 1)  # 前1期收盘价
    TH = np.maximum(HIGH, ref_close_1)  # TH = MAX(HIGH, REF(CLOSE,1))
    TL = np.minimum(LOW, ref_close_1)   # TL = MIN(LOW, REF(CLOSE,1))
    TR = TH - TL                        # 真实波动范围
    XR = CLOSE - TL                     # 上涨幅度
    
    # 4. 计算各周期的SUM(XR)和SUM(TR)（滚动求和）
    sum_XR_N1 = rolling_sum(XR, N1)
    sum_TR_N1 = rolling_sum(TR, N1)
    
    sum_XR_N2 = rolling_sum(XR, N2)
    sum_TR_N2 = rolling_sum(TR, N2)
    
    sum_XR_N3 = rolling_sum(XR, N3)
    sum_TR_N3 = rolling_sum(TR, N3)
    
    # 5. 安全计算XRM、XRN、XRO（过滤分母为0或NaN的情况）
    epsilon = 1e-10  # 处理浮点数精度误差
    # 计算XRM（短周期比率）
    valid_N1 = (sum_TR_N1 > epsilon) & ~np.isnan(sum_TR_N1) & ~np.isnan(sum_XR_N1)
    XRM = np.where(valid_N1, sum_XR_N1 / sum_TR_N1, np.nan)
    
    # 计算XRN（中周期比率）
    valid_N2 = (sum_TR_N2 > epsilon) & ~np.isnan(sum_TR_N2) & ~np.isnan(sum_XR_N2)
    XRN = np.where(valid_N2, sum_XR_N2 / sum_TR_N2, np.nan)
    
    # 计算XRO（长周期比率）
    valid_N3 = (sum_TR_N3 > epsilon) & ~np.isnan(sum_TR_N3) & ~np.isnan(sum_XR_N3)
    XRO = np.where(valid_N3, sum_XR_N3 / sum_TR_N3, np.nan)
    
    # 6. 计算UOS（加权合成）
    denominator_uos = N1*N2 + N1*N3 + N2*N3  # 权重分母（常数，非零）
    numerator_uos = XRM * N2*N3 + XRN * N1*N3 + XRO * N1*N2
    UOS = 100 * numerator_uos / denominator_uos
    
    return pd.Series(UOS, index=CLOSE.index)


def DZRSI(CLOSE, N=14, M=3, PARAM=2):  # 修复版动态RSI（DZRSI）
    """
    计算动态RSI指标的上轨（RSI_UPPER）和下轨（RSI_LOWER），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N: RSI的计算周期（默认14）
    :param M: 移动平均（MA）和标准差（STD）的周期（默认3）
    :param PARAM: 标准差倍数（默认2）
    :return: RSI_UPPER序列、RSI_LOWER序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def SMA(X, window, min_periods=1):
        """简单移动平均（SMA）：滚动窗口平均，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    def MA(X, window):
        """移动平均（等同于SMA，确保与指标定义一致）"""
        return SMA(X, window, min_periods=window)
    
    def STD(X, window):
        """滚动标准差：计算window期内的标准差，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=window).std()
    
    # 3. 计算RSI的分子和分母
    price_change = CLOSE - REF(CLOSE, 1)  # 每日价格变动（涨为正，跌为负）
    up_change = np.maximum(price_change, 0)  # 上涨幅度（下跌时取0）
    abs_change = np.abs(price_change)        # 价格变动绝对值
    
    # 分子：N期上涨幅度的SMA
    sma_up = SMA(up_change, N, min_periods=N)
    # 分母：N期价格变动绝对值的SMA
    sma_abs = SMA(abs_change, N, min_periods=N)
    
    # 4. 安全计算RSI：过滤分母为0或NaN的情况
    epsilon = 1e-10  # 处理浮点数精度误差
    # 有效性条件：分母非0+非NaN + 分子非NaN
    valid = (
        (sma_abs > epsilon) & ~np.isnan(sma_abs) &
        ~np.isnan(sma_up)
    )
    
    # 特殊处理：N期无波动时，RSI理论上为50（中性）
    RSI = np.where(
        valid,
        (sma_up / sma_abs) * 100,
        50.0  # 无价格变动时视为中性，RSI=50
    )
    RSI = pd.Series(RSI, index=CLOSE.index)
    
    # 5. 计算RSI_MIDDLE（RSI的M期移动平均）和标准差
    RSI_MIDDLE = MA(RSI, M)  # RSI的M期平均
    rsi_std = STD(RSI, M)    # RSI的M期标准差
    
    # 6. 计算上轨和下轨（处理标准差为NaN的情况）
    RSI_UPPER = RSI_MIDDLE + PARAM * rsi_std
    RSI_LOWER = RSI_MIDDLE - PARAM * rsi_std
    
    return RSI_UPPER, RSI_LOWER


def DZCCI(CLOSE, HIGH, LOW, N=40, M=3, PARAM=2):  # 动态CCI
    """TP=(HIGH+LOW+CLOSE)/3
       CCI=(TP-MA(TP,N))/(0.015*AVEDEV(TP,N))
       CCI_MIDDLE=MA(CCI,M)
       CCI_UPPER=CCI_MIDDLE+PARAM*STD(CCI,M)
       CCI_LOWER=CCI_MIDDLE-PARAM*STD(CCI,M)"""
    TP = (HIGH + LOW + CLOSE) / 3
    CCI = (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))
    CCI_MIDDLE = MA(CCI, M)
    return CCI_MIDDLE + PARAM * STD(CCI, M), CCI_MIDDLE - PARAM * STD(CCI, M)


def CMF(CLOSE, HIGH, LOW, VOLUME, N=20):  # 修复版蔡金货币流量（CMF）
    """
    计算蔡金货币流量指标（CMF），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param VOLUME: 成交量序列
    :param N: 滚动计算周期（默认20）
    :return: CMF指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    for s in [CLOSE, HIGH, LOW, VOLUME]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            VOLUME = pd.Series(VOLUME)
            break
    
    # 2. 安全计算CLV：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    clv_numerator = 2 * CLOSE - LOW - HIGH  # CLV分子
    clv_denominator = HIGH - LOW            # CLV分母（当天价格波动范围）
    
    # 有效性条件：分母非0+非NaN + 分子非NaN（避免无效运算）
    valid_clv = (
        (clv_denominator > epsilon) & ~np.isnan(clv_denominator) &
        ~np.isnan(clv_numerator)
    )
    
    # 仅在有效时计算CLV，无效时设为NaN（避免除零）
    CLV = np.where(valid_clv, clv_numerator / clv_denominator, np.nan)
    CLV = pd.Series(CLV, index=CLOSE.index)  # 保持索引一致
    
    # 3. 实现正确的滚动SUM（N期内求和，而非全局求和）
    def rolling_sum(X, window):
        """N期滚动求和：仅窗口内有window个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).sum()
    
    # 4. 计算CMF的分子和分母（滚动求和）
    cmf_numerator = rolling_sum(CLV * VOLUME, N)  # 分子：CLV×成交量的N期滚动和
    cmf_denominator = rolling_sum(VOLUME, N)      # 分母：成交量的N期滚动和
    
    # 5. 处理CMF的二次除零（避免成交量滚动和为0的极端情况）
    valid_cmf = (
        (cmf_denominator > epsilon) & ~np.isnan(cmf_denominator) &
        ~np.isnan(cmf_numerator)
    )
    CMF = np.where(valid_cmf, cmf_numerator / cmf_denominator, np.nan)
    
    return pd.Series(CMF, index=CLOSE.index)


def PPO(CLOSE, N1=12, N2=26, N3=9):  # PPO指标
    """PPO=(EMA(CLOSE,N1)-EMA(CLOSE,N2))/EMA(CLOSE,N2)
       PPO_SIGNAL=EMA(PPO,N3)"""
    PPO = (EMA(CLOSE, N1) - EMA(CLOSE, N2)) / EMA(CLOSE, N2)
    return PPO, EMA(PPO, N3)


def RWI(CLOSE, HIGH, LOW, N=14):  # 随机漫步指标
    """TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-LOW))
       ATR=MA(TR,N)
       RWIH=(HIGH-REF(LOW,1))/(ATR*SQRT(N))
       RWIL=(REF(HIGH,1)-LOW)/(ATR*SQRT(N))"""
    temp_max = MAX(ABS(HIGH - LOW), ABS(HIGH - REF(CLOSE, 1)))
    TR = MAX(temp_max, ABS(REF(CLOSE, 1) - LOW))
    ATR = MA(TR, N)
    RWIH = (HIGH - REF(LOW, 1)) / (ATR * np.sqrt(N))
    RWIL = (REF(HIGH, 1) - LOW) / (ATR * np.sqrt(N))
    return RWIH, RWIL


def ATR(HIGH, LOW, CLOSE, N=14):  # 真实波动幅度均值
    """TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-LOW))
       ATR=MA(TR,N)"""
    temp_max = MAX(ABS(HIGH - LOW), ABS(HIGH - REF(CLOSE, 1)))
    TR = MAX(temp_max, ABS(REF(CLOSE, 1) - LOW))
    return MA(TR, N)


def WAD(CLOSE, HIGH, LOW, VOLUME, N=20):  # 威廉累积离散量
    """TRH=MAX(HIGH,REF(CLOSE,1))
       TRL=MIN(LOW,REF(CLOSE,1))
       AD=IF(CLOSE>REF(CLOSE,1),CLOSE-TRL,IF(CLOSE<REF(CLOSE,1),CLOSE-TRH,0))
       WAD=CUMSUM(AD)
       WADMA=MA(WAD,N)"""
    TRH = MAX(HIGH, REF(CLOSE, 1))
    TRL = MIN(LOW, REF(CLOSE, 1))
    AD = IF(CLOSE > REF(CLOSE, 1), CLOSE - TRL, IF(CLOSE < REF(CLOSE, 1), CLOSE - TRH, 0))
    WAD = CUM_SUM(AD)
    return WAD, MA(WAD, N)


def KST(CLOSE, N1=10, N2=10, N3=10, N4=10, M=9):  # KST指标
    """ROC1=(CLOSE-REF(CLOSE,10))/REF(CLOSE,10)
       ROC2=(CLOSE-REF(CLOSE,20))/REF(CLOSE,20)
       ROC3=(CLOSE-REF(CLOSE,30))/REF(CLOSE,30)
       ROC4=(CLOSE-REF(CLOSE,40))/REF(CLOSE,40)
       KST=MA(ROC1+2*ROC2+3*ROC3+4*ROC4,M)"""
    ROC1 = (CLOSE - REF(CLOSE, 10)) / REF(CLOSE, 10)
    ROC2 = (CLOSE - REF(CLOSE, 20)) / REF(CLOSE, 20)
    ROC3 = (CLOSE - REF(CLOSE, 30)) / REF(CLOSE, 30)
    ROC4 = (CLOSE - REF(CLOSE, 40)) / REF(CLOSE, 40)
    return MA(ROC1 + 2 * ROC2 + 3 * ROC3 + 4 * ROC4, M)


def VI(CLOSE, HIGH, LOW, N=40):  # VI指标
    """TR=MAX(ABS(HIGH-LOW),ABS(LOW-REF(CLOSE,1)),ABS(HIGH-REF(CLOSE,1)))
       VMPOS=ABS(HIGH-REF(LOW,1))
       VMNEG=ABS(LOW-REF(HIGH,1))
       SUMPOS=SUM(VMPOS,N)
       SUMNEG=SUM(VMNEG,N)
       TRSUM=SUM(TR,N)
       VI+=SUMPOS/TRSUM
       VI-=SUMNEG/TRSUM"""
    temp_max = MAX(ABS(HIGH - LOW), ABS(HIGH - REF(CLOSE, 1)))
    TR = MAX(temp_max, ABS(REF(CLOSE, 1) - LOW))
    VMPOS = ABS(HIGH - REF(LOW, 1))
    VMNEG = ABS(LOW - REF(HIGH, 1))
    SUMPOS = SUM(VMPOS, N)
    SUMNEG = SUM(VMNEG, N)
    TRSUM = SUM(TR, N)
    return SUMPOS / TRSUM, SUMNEG / TRSUM


def DMA(CLOSE, N=12):  # 动态移动平均
    """DMA=2/(N+1)*CLOSE+(N-1)/(N+1)*REF(DMA,1)"""
    if isinstance(CLOSE, np.ndarray):
        CLOSE = pd.Series(CLOSE)

        # 初始化 DMA 序列，第一个值等于 CLOSE 的第一个值
    dma = pd.Series(index=CLOSE.index)
    dma.iloc[0] = CLOSE.iloc[0]

    # 从第二个值开始迭代计算 DMA
    for i in range(1, len(CLOSE)):
        dma.iloc[i] = 2 / (N + 1) * CLOSE.iloc[i] + (N - 1) / (N + 1) * dma.iloc[i - 1]

    return dma


def MICD(CLOSE, N=20, N1=10, N2=20, M=10):  # MICD指标
    """MI=CLOSE-REF(CLOSE,1)
       MTMMA=SMA(MI,N,1)
       DIF=MA(REF(MTMMA,1),N1)-MA(REF(MTMMA,1),N2)
       MICD=SMA(DIF,M,1)"""
    MI = CLOSE - REF(CLOSE, 1)
    MTMMA = SMA(MI, N, 1)
    DIF = MA(REF(MTMMA, 1), N1) - MA(REF(MTMMA, 1), N2)
    return SMA(DIF, M, 1)


def PMO(CLOSE, N1=10, N2=40, N3=20):  # PMO指标
    """ROC=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*100
       ROC_MA=DMA(ROC,2/N1)
       PMO=DMA(ROC_MA*10,2/N2)
       PMO_SIGNAL=DMA(PMO,2/(N3+1))"""
    # 1. 标准化输入为pandas.Series，避免数组操作导致的NaN传递
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))  # 给索引确保对齐

    # 2. 计算ROC：(当前收盘价 - 前1期收盘价)/前1期收盘价 * 100（正确实现REF(CLOSE,1)）
    ref_close_1 = CLOSE.shift(1)  # 替代REF(CLOSE,1)，第一行是NaN，第2行起有值
    ROC = (CLOSE - ref_close_1) / ref_close_1 * 100  # 仅第一行NaN，后续359行有值

    # 3. 实现正确的DMA（动态移动平均=指数移动平均EMA）
    def DMA(X, alpha):
        """
        正确实现DMA(X,A) = 指数移动平均（EMA）
        X: 输入序列（pandas.Series）
        alpha: 平滑系数（A=2/N，对应EMA的span参数：span = 2/alpha - 1）
        """
        # 计算EMA的span参数（EMA(span)的平滑系数=2/(span+1)，与DMA的A=2/N对应）
        span = (2 / alpha) - 1
        # 用EMA计算DMA，ignore_na=False确保不跳过NaN（避免初始值异常）
        # min_periods=1：只要有1个有效值就计算，初始NaN会被后续值覆盖
        dma_series = X.ewm(span=span, min_periods=1, ignore_na=False).mean()
        return dma_series

    # 4. 逐步计算PMO链条（每一步都不会全为NaN）
    # ROC_MA = DMA(ROC, 2/N1)：ROC第一行NaN，第二行起ROC_MA有值
    ROC_MA = DMA(ROC, alpha=2 / N1)
    # PMO = DMA(ROC_MA*10, 2/N2)：基于ROC_MA，第二行起有值
    PMO = DMA(ROC_MA * 10, alpha=2 / N2)
    # PMO_SIGNAL = DMA(PMO, 2/(N3+1))：基于PMO，第二行起有值
    PMO_SIGNAL = DMA(PMO, alpha=2 / (N3 + 1))

    return PMO, PMO_SIGNAL


def RCCD(CLOSE, N=40, N1=20, N2=40, M=40):  # RCCD指标
    """RC=CLOSE/REF(CLOSE,N)
       ARC1=SMA(REF(RC,1),M,1)
       DIF=MA(REF(ARC1,1),N1)-MA(REF(ARC1,1),N2)
       RCCD=SMA(DIF,M,1)"""
    RC = CLOSE / REF(CLOSE, N)
    ARC1 = SMA(REF(RC, 1), M, 1)
    DIF = MA(REF(ARC1, 1), N1) - MA(REF(ARC1, 1), N2)
    return SMA(DIF, M, 1)


def KAMA(CLOSE, N=10, N1=2, N2=30):
    """
    计算KAMA（自适应移动平均）指标，修复全为NaN的问题
    CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    N: 计算波动率的窗口大小（默认10）
    N1: 快速平滑参数（默认2）
    N2: 慢速平滑参数（默认30）
    返回：KAMA序列（pandas.Series）
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))

    # 2. 计算DIRECTION和VOLATILITY（用pandas原生函数替代REF和SUM）
    # 前N期收盘价（替代REF(CLOSE, N)）
    ref_close_n = CLOSE.shift(N)
    # 方向：当前收盘价 - 前N期收盘价（取绝对值）
    direction = abs(CLOSE - ref_close_n)

    # 单日价格变化（替代REF(CLOSE, 1)）
    daily_change = abs(CLOSE - CLOSE.shift(1))
    # 波动率：过去N期单日变化的滚动和（替代SUM(..., N)）
    # min_periods=N：确保只有N期数据足够时才计算，前N-1期为NaN
    volatility = daily_change.rolling(window=N, min_periods=N).sum()

    # 3. 计算ER（效率比率），处理volatility=0的情况（避免除零）
    epsilon = 1e-10  # 极小值，避免除零
    # 当volatility为0或NaN时，ER设为0（合理默认值）
    ER = np.where(
        (volatility > epsilon) & ~np.isnan(volatility) & ~np.isnan(direction),
        direction / volatility,
        0.0  # 替代NaN，避免后续计算中断
    )
    ER = pd.Series(ER, index=CLOSE.index)  # 转为Series保持索引

    # 4. 计算平滑常数
    fastest = 2 / (N1 + 1)
    slowest = 2 / (N2 + 1)
    SC = (ER * (fastest - slowest) + slowest) ** 2  # 平滑系数的平方

    # 5. 初始化KAMA序列，前N期数据不足，用NaN填充
    kama = pd.Series(index=CLOSE.index, dtype='float64')
    # 第N期的初始值：用第N期收盘价（此时已有足够数据计算volatility）
    if len(CLOSE) > N:
        kama.iloc[N] = CLOSE.iloc[N]  # 第一个有效初始值

    # 6. 从第N+1期开始迭代计算（确保前期数据有效）
    for i in range(N + 1, len(CLOSE)):
        # 用前一期的KAMA值计算当前值（若前一期为NaN，当前也设为NaN）
        if pd.notna(kama.iloc[i - 1]):
            kama.iloc[i] = SC.iloc[i] * CLOSE.iloc[i] + (1 - SC.iloc[i]) * kama.iloc[i - 1]

    return kama


def AWS(CLOSE, N=20):  # AWS指标
    """AWS=EMA(EMA(EMA(CLOSE,N),N),N)"""
    return EMA(EMA(EMA(CLOSE, N), N), N)


def ARBR(CLOSE, HIGH, LOW, OPEN, N=26):  # ARBR指标
    """AR=SUM(HIGH-OPEN,N)/SUM(OPEN-LOW,N)*100
       BR=SUM(HIGH-REF(CLOSE,1),N)/SUM(REF(CLOSE,1)-LOW,N)*100"""
    AR = SUM(HIGH - OPEN, N) / SUM(OPEN - LOW, N) * 100
    BR = SUM(HIGH - REF(CLOSE, 1), N) / SUM(REF(CLOSE, 1) - LOW, N) * 100
    return AR, BR


def ADXR(ADX, N=6):  # ADXR指标
    """ADXR=(ADX+REF(ADX,N))/2"""
    return (ADX + REF(ADX, N)) / 2


def SMI(CLOSE, HIGH, LOW, N1=20, N2=20, N3=20):  # SMI指标
    """M=(MAX(HIGH,N1)+MIN(LOW,N1))/2
       D=CLOSE-M
       DS=EMA(EMA(D,N2),N2)
       DHL=EMA(EMA(MAX(HIGH,N1)-MIN(LOW,N1),N2),N2)
       SMI=100*DS/DHL
       SMIMA=MA(SMI,N3)"""
    M = (HHV(HIGH, N1) + LLV(LOW, N1)) / 2
    D = CLOSE - M
    DS = EMA(EMA(D, N2), N2)
    DHL = EMA(EMA(HHV(HIGH, N1) - LLV(LOW, N1), N2), N2)
    SMI = 100 * DS / DHL
    return SMI, MA(SMI, N3)


def SI(CLOSE, HIGH, LOW, OPEN, N=20, M=20):  # 修复版SI指标（含ASI及ASIMA）
    """
    计算摆动指标SI、累积线ASI及平均线ASIMA，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param OPEN: 开盘价序列
    :param N: 价格波动下限参数（默认20）
    :param M: ASIMA移动平均窗口（默认20）
    :return: ASI序列、ASIMA序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐（避免数组索引错位）
    for s in [CLOSE, HIGH, LOW, OPEN]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            OPEN = pd.Series(OPEN)
            break
    
    # 2. 用原生函数替代自定义的REF/ABS/MAX/IF（避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    # 3. 计算中间变量A/B/C/D
    ref_close1 = REF(CLOSE, 1)  # 前1期收盘价
    ref_low1 = REF(LOW, 1)      # 前1期最低价
    ref_open1 = REF(OPEN, 1)    # 前1期开盘价
    
    A = np.abs(HIGH - ref_close1)  # 绝对值用np.abs
    B = np.abs(LOW - ref_close1)
    C = np.abs(HIGH - ref_low1)
    D = np.abs(ref_close1 - ref_open1)
    
    # 4. 计算K和R（用np.maximum替代MAX，np.where替代IF）
    K = np.maximum(A, B)  # 取A和B的最大值
    
    # 计算R的条件判断（三层逻辑用嵌套np.where实现）
    R1 = A + 0.5 * B + 0.25 * D
    R2 = B + 0.5 * A + 0.25 * D
    R3 = C + 0.25 * D
    
    cond1 = (C >= A) & (C >= B)  # 条件1：C是A/B/C中的最大值
    cond2 = ~cond1 & (A >= B)    # 条件2：A≥B（且不满足条件1）
    R = np.where(cond1, R3, np.where(cond2, R1, R2))  # 最终R值
    
    # 5. 处理除法无效值：过滤R=0/NaN、第二个分母异常的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    
    # 计算第二个分母：MAX(HIGH-LOW, N)（确保至少为N，避免异常）
    price_range = HIGH - LOW
    denominator2 = np.maximum(price_range, N)  # 原生np.maximum确保逻辑正确
    
    # 计算SI的分子（可简化表达式，不影响结果）
    si_numerator = 50 * (
        (CLOSE - ref_close1) + 
        (ref_close1 - ref_open1) + 
        0.5 * (CLOSE - OPEN)
    )
    
    # 有效性总条件：R有效 + 分母2有效 + 分子/K有效
    valid = (
        (R > epsilon) & ~np.isnan(R) &          # R非0且非NaN
        (denominator2 > epsilon) & ~np.isnan(denominator2) &  # 分母2有效
        ~np.isnan(si_numerator) & ~np.isnan(K)  # 分子和K非NaN
    )
    
    # 安全计算SI：仅在有效时执行除法，否则为NaN
    SI = np.where(
        valid,
        si_numerator / R * K / denominator2,
        np.nan
    )
    
    # 6. 计算ASI（累积和）和ASIMA（移动平均）
    ASI = pd.Series(SI).cumsum()  # 替代自定义CUM_SUM（原生cumsum更稳健）
    
    def MA(X, window):
        """简单移动平均（处理NaN，避免无效值扩散）"""
        return pd.Series(X).rolling(window=window, min_periods=1).mean()
    
    ASIMA = MA(ASI, M)
    
    return ASI, ASIMA


def DO(CLOSE, N=20):  # 修复版DO指标
    """
    计算DO指标（对RSI进行两次EMA平滑），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N: 计算周期（默认20，用于RSI和两次EMA）
    :return: DO指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def SMA(X, window, min_periods=1):
        """简单移动平均（SMA）：滚动窗口平均，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    def EMA(X, window, min_periods=1):
        """指数移动平均（EMA）：符合金融指标定义的平滑逻辑（adjust=False）"""
        return X.ewm(span=window, min_periods=min_periods, adjust=False).mean()
    
    # 3. 计算RSI的核心变量：价格变动、上涨幅度、变动绝对值
    ref_close1 = REF(CLOSE, 1)  # 前1期收盘价
    price_change = CLOSE - ref_close1  # 每日价格变动（涨为正，跌为负）
    up_change = np.maximum(price_change, 0)  # 上涨幅度（下跌时取0）
    abs_change = np.abs(price_change)        # 价格变动绝对值
    
    # 4. 计算RSI的分子（上涨动能SMA）和分母（总波动SMA）
    sma_up = SMA(up_change, N, min_periods=N)  # N期上涨动能的SMA
    sma_abs = SMA(abs_change, N, min_periods=N)  # N期总波动的SMA
    
    # 5. 安全计算RSI：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    # 有效性条件：分母非0+非NaN + 分子非NaN
    valid_rsi = (
        (sma_abs > epsilon) & ~np.isnan(sma_abs) &
        ~np.isnan(sma_up)
    )
    
    # 特殊处理：N期无波动时，RSI理论为50（中性），避免除零
    RSI = np.where(
        valid_rsi,
        (sma_up / sma_abs) * 100,
        50.0
    )
    RSI = pd.Series(RSI, index=CLOSE.index)
    
    # 6. 计算DO指标：对RSI进行两次EMA平滑
    ema1 = EMA(RSI, N, min_periods=N)  # 第一次EMA
    DO = EMA(ema1, N, min_periods=N)   # 第二次EMA
    
    return DO


def DBCD(CLOSE, N=5, M=16, T=17):  # DBCD指标
    """BIAS=(CLOSE-MA(CLOSE,N))/MA(CLOSE,N)*100
       BIAS_DIF=BIAS-REF(BIAS,M)
       DBCD=SMA(BIAS_DIF,T,1)"""
    BIAS = (CLOSE - MA(CLOSE, N)) / MA(CLOSE, N) * 100
    BIAS_DIF = BIAS - REF(BIAS, M)
    return SMA(BIAS_DIF, T, 1)

def CV(CLOSE, HIGH, LOW, N=10):  # 修复版CV指标（价格波动变异率）
    """
    计算CV指标，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param N: 计算周期（默认10）
    :return: CV指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    for s in [HIGH, LOW]:
        if not isinstance(s, pd.Series):
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break

    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)，前n期为NaN"""
        return X.shift(n)

    def EMA(X, window, min_periods=1):
        """指数移动平均（EMA）：符合金融指标定义，adjust=False确保平滑逻辑正确"""
        return X.ewm(span=window, min_periods=min_periods, adjust=False).mean()

    # 3. 计算H_L_EMA（价格波动范围的N期EMA）
    price_range = HIGH - LOW  # 每日价格波动范围（最高价-最低价，非负）
    H_L_EMA = EMA(price_range, N, min_periods=N)  # N期EMA，前N-1期为NaN

    # 4. 计算分母：前N期的H_L_EMA
    ref_hl_ema_n = REF(H_L_EMA, N)  # 前N期的H_L_EMA值

    # 5. 安全计算CV：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    # 有效性条件：分母非0+非NaN + 分子非NaN
    valid = (
            (np.abs(ref_hl_ema_n) > epsilon) & ~np.isnan(ref_hl_ema_n) &
            ~np.isnan(H_L_EMA)
    )

    # 分母无效时（为0或NaN），CV设为NaN（无意义）
    CV = np.where(
        valid,
        (H_L_EMA - ref_hl_ema_n) / ref_hl_ema_n * 100,
        np.nan
    )

    return pd.Series(CV, index=HIGH.index)


# 表35:价格反转指标

def KDJ(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):  # 修复版KDJ指标
    """
    计算KDJ指标，修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param N: RSV窗口（默认9）
    :param M1: K值SMA窗口（默认3）
    :param M2: D值SMA窗口（默认3）
    :return: K、D、J序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    for s in [CLOSE, HIGH, LOW]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break
    
    # 2. 实现正确的滚动LLV（N期最低价）和HHV（N期最高价）
    def LLV(X, window):
        """N期滚动最小值，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=window).min()
    
    def HHV(X, window):
        """N期滚动最大值，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=window).max()
    
    # 3. 计算N期滚动最值（避免全局最值或数据不足问题）
    LOW_N = LLV(LOW, N)   # N期内最低价
    HIGH_N = HHV(HIGH, N) # N期内最高价
    
    # 4. 安全计算RSV：过滤分母为0或NaN的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    price_range = HIGH_N - LOW_N  # 分母：N期价格波动范围
    
    # 有效性条件：分母非0+非NaN + 收盘价在N期区间内（避免异常值）
    valid = (
        (price_range > epsilon) & ~np.isnan(price_range) &
        (CLOSE >= LOW_N - epsilon) & (CLOSE <= HIGH_N + epsilon) &
        ~np.isnan(CLOSE)
    )
    
    # 仅在有效时计算RSV，无效时设为NaN（避免除零）
    RSV = np.where(
        valid,
        (CLOSE - LOW_N) / price_range * 100,
        np.nan
    )
    RSV = pd.Series(RSV, index=CLOSE.index)  # 保持索引一致
    
    # 5. 实现正确的SMA（简单移动平均），处理NaN
    def SMA(X, window, min_periods=1):
        """滚动窗口简单移动平均，自动忽略NaN"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    # 6. 计算K、D、J值
    K = SMA(RSV, M1, 1)  # K = SMA(RSV, M1, 1)
    D = SMA(K, M2, 1)    # D = SMA(K, M2, 1)
    J = 3 * K - 2 * D    # J = 3K - 2D
    
    return K, D, J


def RMI(CLOSE, N=7):  # RMI指标
    """CLOSEUP=IF(CLOSE>REF(CLOSE,4),CLOSE-REF(CLOSE,4),0)
       CLOSEDOWN=IF(CLOSE<REF(CLOSE,4),REF(CLOSE,4)-CLOSE,0)
       CLOSEUP_MA=SMA(CLOSEUP,N,1)
       CLOSEDOWN_MA=SMA(CLOSEDOWN,N,1)
       RMI=100*CLOSEUP_MA/(CLOSEUP_MA+CLOSEDOWN_MA)"""
    # 确保输入是pandas.Series，方便移位和移动平均计算
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE)

    # 计算REF(CLOSE, 4)：前4期收盘价（替代原REF函数）
    ref_close = CLOSE.shift(4)

    # 计算CLOSEUP和CLOSEDOWN（替代原IF函数）
    CLOSEUP = np.where(CLOSE > ref_close, CLOSE - ref_close, 0)
    CLOSEDOWN = np.where(CLOSE < ref_close, ref_close - CLOSE, 0)

    # 计算SMA（简单移动平均，替代原SMA函数）
    # 使用pandas的rolling窗口，min_periods=N确保前N-1个值为NaN（避免无效平均）
    CLOSEUP_MA = pd.Series(CLOSEUP).rolling(window=N, min_periods=N).mean()
    CLOSEDOWN_MA = pd.Series(CLOSEDOWN).rolling(window=N, min_periods=N).mean()

    # 计算分母并强化有效性检查
    denominator = CLOSEUP_MA + CLOSEDOWN_MA
    epsilon = 1e-10

    # 有效条件：分母非NaN且绝对值大于epsilon（避免0或接近0）
    valid = ~np.isnan(denominator) & (np.abs(denominator) > epsilon)

    # 计算RMI，无效情况返回NaN
    RMI = np.where(valid, 100 * CLOSEUP_MA / denominator, np.nan)

    return RMI


def SKDJ(CLOSE, HIGH, LOW, N=9, M=3):  # 修复版SKDJ指标（慢速随机指标）
    """
    计算慢速随机指标（SKDJ），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param N: RSV的滚动窗口（默认9，N期最值）
    :param M: K/D值的SMA窗口（默认3，平滑周期）
    :return: K值序列、D值序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐（避免滚动计算错位）
    for s in [CLOSE, HIGH, LOW]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break
    
    # 2. 实现正确的LLV（滚动N期最小值）和HHV（滚动N期最大值）
    def LLV(X, window):
        """滚动N期最小值：仅当窗口内有N个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).min()
    
    def HHV(X, window):
        """滚动N期最大值：仅当窗口内有N个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).max()
    
    # 3. 计算N期滚动最值（避免全局最值或数据不足问题）
    LOW_N = LLV(LOW, N)   # N期内最低价
    HIGH_N = HHV(HIGH, N) # N期内最高价
    
    # 4. 安全计算RSV：过滤分母为0或NaN的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    price_range = HIGH_N - LOW_N  # 分母：N期价格波动范围
    
    # 有效性条件：分母非0+非NaN + 收盘价在N期区间内（避免异常值）
    valid = (
        (price_range > epsilon) & ~np.isnan(price_range) &
        (CLOSE >= LOW_N - epsilon) & (CLOSE <= HIGH_N + epsilon) &
        ~np.isnan(CLOSE)
    )
    
    # 仅在有效时计算RSV，无效时设为NaN（彻底避免除零）
    RSV = np.where(
        valid,
        (CLOSE - LOW_N) / price_range * 100,
        np.nan
    )
    RSV = pd.Series(RSV, index=CLOSE.index)  # 保持索引与原始数据一致
    
    # 5. 实现正确的SMA（简单移动平均），处理中间NaN值
    def SMA(X, window, min_periods=1):
        """滚动窗口简单移动平均：自动忽略NaN，确保平滑逻辑正确"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    # 6. 计算SKDJ的K、D值
    K = SMA(RSV, M, 1)  # K = SMA(RSV, M, 1)（慢速K线）
    D = SMA(K, M, 1)    # D = SMA(K, M, 1)（慢速D线）
    
    return K, D


def CCI(CLOSE, HIGH, LOW, N=14):  # CCI指标
    """TP=(HIGH+LOW+CLOSE)/3
       CCI=(TP-MA(TP,N))/(0.015*AVEDEV(TP,N))"""
    TP = (HIGH + LOW + CLOSE) / 3
    return (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))


def RSI(CLOSE, N=24):  # 修复版RSI指标
    """
    计算相对强弱指数（RSI），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N: 计算周期（默认24）
    :return: RSI指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def SMA(X, window, min_periods=1):
        """简单移动平均（SMA）：滚动窗口平均，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    # 3. 计算价格变动、上涨幅度、变动绝对值
    price_change = CLOSE - REF(CLOSE, 1)  # 每日价格变动（涨为正，跌为负）
    up_change = np.maximum(price_change, 0)  # 上涨幅度（下跌时取0）
    abs_change = np.abs(price_change)        # 价格变动绝对值
    
    # 4. 计算CLOSEUP（上涨动能的SMA）和CLOSEDOWN（总波动动能的SMA）
    CLOSEUP = SMA(up_change, N, min_periods=N)  # N期上涨幅度的移动平均
    CLOSEDOWN = SMA(abs_change, N, min_periods=N)  # N期总波动的移动平均
    
    # 5. 安全计算RSI：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差
    # 有效性条件：分母非0+非NaN + 分子非NaN
    valid = (
        (CLOSEDOWN > epsilon) & ~np.isnan(CLOSEDOWN) &
        ~np.isnan(CLOSEUP)
    )
    
    # 特殊处理：N期无波动时，RSI理论上为50（中性，涨跌平衡）
    RSI = np.where(
        valid,
        100 * (CLOSEUP / CLOSEDOWN),
        50.0  # 无价格变动时，强弱平衡，RSI=50
    )
    
    return pd.Series(RSI, index=CLOSE.index)


def ROC(CLOSE, N=100):  # ROC指标
    """ROC=(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)*100"""
    return (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N) * 100


def WR(CLOSE, HIGH, LOW, N=10, N1=6):  # 修复版W&R威廉指标
    """
    计算威廉指标（W&R），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param N: WR的计算周期（默认10）
    :param N1: WR1的计算周期（默认6，短周期）
    :return: WR序列、WR1序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算时索引对齐
    for s in [CLOSE, HIGH, LOW]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            break
    
    # 2. 实现正确的滚动HHV（最大值）和LLV（最小值）
    def rolling_HHV(X, window):
        """N期滚动最高价：仅窗口内有window个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).max()
    
    def rolling_LLV(X, window):
        """N期滚动最低价：仅窗口内有window个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).min()
    
    # 3. 计算两个周期的滚动最值（避免全局最值或数据不足问题）
    # WR周期（N=10）
    high_N = rolling_HHV(HIGH, N)
    low_N = rolling_LLV(LOW, N)
    # WR1周期（N1=6）
    high_N1 = rolling_HHV(HIGH, N1)
    low_N1 = rolling_LLV(LOW, N1)
    
    # 4. 定义通用函数：安全计算WR（避免重复代码）
    def calculate_WR(high_period, low_period, close):
        """
        安全计算单周期WR值
        :param high_period: 周期内滚动最高价
        :param low_period: 周期内滚动最低价
        :param close: 收盘价
        :return: 单周期WR序列
        """
        epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
        denominator = high_period - low_period  # 周期价格波动范围（分母）
        numerator = high_period - close         # 周期最高价-收盘价（分子）
        
        # 有效性条件：分母非0+非NaN + 分子非NaN（避免无效运算）
        valid = (
            (denominator > epsilon) & ~np.isnan(denominator) &
            ~np.isnan(numerator) & ~np.isnan(close)
        )
        
        # 仅在有效时计算WR，无效时设为NaN（避免除零）
        wr = np.where(
            valid,
            (numerator / denominator) * 100,
            np.nan
        )
        return pd.Series(wr, index=close.index)
    
    # 5. 分别计算WR和WR1
    WR = calculate_WR(high_N, low_N, CLOSE)
    WR1 = calculate_WR(high_N1, low_N1, CLOSE)
    
    return WR, WR1


def STC(CLOSE, HIGH, LOW, N1=23, N2=50, N=40):  # STC指标
    """MACD=EMA(CLOSE,N1)-EMA(CLOSE,N2)
       V1=LLV(MACD,N)
       V2=HHV(MACD,N)-V1
       FK=IF(V2>0,(MACD-V1)/V2*100,REF(FK,1))
       FD=SMA(FK,N,1)
       V3=LLV(FD,N)
       V4=HHV(FD,N)-V3
       SK=IF(V4>0,(FD-V3)/V4*100,REF(SK,1))
       STC=SMA(SK,N,1)"""
    MACD = EMA(CLOSE, N1) - EMA(CLOSE, N2)
    V1 = LLV(MACD, N)
    V2 = HHV(MACD, N) - V1
    FK = np.full_like(MACD, np.nan)  # 初始化FK为全NaN数组
    FK = IF(V2 > 0, (MACD - V1) / V2 * 100, REF(FK, 1))
    FD = SMA(FK, N, 1)
    V3 = LLV(FD, N)
    V4 = HHV(FD, N) - V3
    SK = np.full_like(FD, np.nan)  # 初始化SK为全NaN数组
    SK = IF(V4 > 0, (FD - V3) / V4 * 100, REF(SK, 1))
    return SMA(SK, N, 1)


def RVI(CLOSE, N1=10, N2=20):  # RVI指标
    """STD=STD(CLOSE,N1)
       USTD=SMA(IF(CLOSE>REF(CLOSE,1),STD,0),N2,1)
       DSTD=SMA(IF(CLOSE<REF(CLOSE,1),STD,0),N2,1)
       RVI=100*USTD/(USTD+DSTD)"""
    # 1. 标准化输入为pandas.Series，方便后续计算
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE)

    # 2. 计算STD(CLOSE, N1)：N1期收盘价标准差（min_periods=N1确保足够数据才计算）
    std_close = CLOSE.rolling(window=N1, min_periods=N1).std()

    # 3. 计算REF(CLOSE, 1)：前1期收盘价（替代原REF函数）
    ref_close_1 = CLOSE.shift(1)

    # 4. 计算USTD和DSTD（替代原IF+SMA逻辑）
    # USTD：上涨周期的标准差移动平均
    ustd_raw = np.where(CLOSE > ref_close_1, std_close, 0)
    USTD = pd.Series(ustd_raw).rolling(window=N2, min_periods=N2).mean()

    # DSTD：下跌周期的标准差移动平均
    dstd_raw = np.where(CLOSE < ref_close_1, std_close, 0)
    DSTD = pd.Series(dstd_raw).rolling(window=N2, min_periods=N2).mean()

    # 5. 强化分母有效性检查，彻底过滤无效值
    denominator = USTD + DSTD
    epsilon = 1e-10
    # 有效条件：分母非NaN + 绝对值大于epsilon（避免0或接近0）
    valid_mask = ~np.isnan(denominator) & (np.abs(denominator) > epsilon)

    # 6. 计算RVI，无效情况返回NaN
    RVI = np.where(valid_mask, 100 * USTD / denominator, np.nan)

    return RVI


def RSIS(CLOSE, N=120, M=20):  # 修复版RSIS指标
    """
    计算RSIS指标及其EMA平滑线（RSISMA），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param N: RSI和RSIS的计算周期（默认120）
    :param M: RSISMA的EMA周期（默认20）
    :return: RSISMA指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    if not isinstance(CLOSE, pd.Series):
        CLOSE = pd.Series(CLOSE, index=range(len(CLOSE)))
    
    # 2. 实现正确的基础函数（替代自定义函数，避免逻辑错误）
    def REF(X, n):
        """取前n期值，对应pandas.shift(n)"""
        return X.shift(n)
    
    def SMA(X, window, min_periods=1):
        """简单移动平均（SMA）：滚动窗口平均，数据不足时返回NaN"""
        return X.rolling(window=window, min_periods=min_periods).mean()
    
    def LLV(X, window):
        """N期滚动最小值：仅窗口完整时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).min()
    
    def HHV(X, window):
        """N期滚动最大值：仅窗口完整时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).max()
    
    def EMA(X, window, min_periods=1):
        """指数移动平均（EMA）：符合金融指标定义的平滑逻辑"""
        return X.ewm(span=window, min_periods=min_periods, adjust=False).mean()
    
    # 3. 计算CLOSE_DIFF_POS（单日上涨动能）
    ref_close1 = REF(CLOSE, 1)  # 前1期收盘价
    # 用np.where实现IF逻辑：收盘价>前一日时取差值，否则为0
    CLOSE_DIFF_POS = np.where(CLOSE > ref_close1, CLOSE - ref_close1, 0)
    CLOSE_DIFF_POS = pd.Series(CLOSE_DIFF_POS, index=CLOSE.index)
    
    # 4. 计算RSI的分子（上涨动能SMA）和分母（总波动SMA）
    price_change_abs = np.abs(CLOSE - ref_close1)  # 每日价格变动绝对值
    sma_up = SMA(CLOSE_DIFF_POS, N, min_periods=N)  # N期上涨动能的SMA
    sma_abs = SMA(price_change_abs, N, min_periods=N)  # N期总波动的SMA
    
    # 5. 安全计算RSI：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差
    valid_rsi = (
        (sma_abs > epsilon) & ~np.isnan(sma_abs) &
        ~np.isnan(sma_up)
    )
    # 无波动时RSI理论为50（中性），避免除零
    RSI = np.where(valid_rsi, (sma_up / sma_abs) * 100, 50.0)
    RSI = pd.Series(RSI, index=CLOSE.index)
    
    # 6. 计算RSIS（RSI的N期归一化），额外处理二次除零风险
    rsi_low = LLV(RSI, N)  # N期内RSI最小值
    rsi_high = HHV(RSI, N)  # N期内RSI最大值
    rsi_range = rsi_high - rsi_low  # 分母：RSI波动范围
    
    valid_rsis = (
        (rsi_range > epsilon) & ~np.isnan(rsi_range) &
        ~np.isnan(RSI) & ~np.isnan(rsi_low)
    )
    # RSI波动范围为0时（RSI恒为50），RSIS设为50（中性）
    RSIS = np.where(
        valid_rsis,
        (RSI - rsi_low) / rsi_range * 100,
        50.0
    )
    RSIS = pd.Series(RSIS, index=CLOSE.index)
    
    # 7. 计算RSISMA（RSIS的EMA平滑）
    RSISMA = EMA(RSIS, M, min_periods=M)
    
    return RSISMA


# 表36:成交量指标

def MAAMT(VOLUME, N=40):  # 成交额移动平均
    """MAAMT=MA(VOLUME,N)"""
    return MA(VOLUME, N)


def SROCVOL(VOLUME, N=20, M=10):  # 平滑成交量ROC
    """EMAP=EMA(VOLUME,N)
       SROCVOL=(EMAP-REF(EMAP,M))/REF(EMAP,M)"""
    EMAP = EMA(VOLUME, N)
    return (EMAP - REF(EMAP, M)) / REF(EMAP, M)


def PVO(VOLUME, N1=12, N2=26):  # 成交量PVO
    """PVO=(EMA(VOLUME,N1)-EMA(VOLUME,N2))/EMA(VOLUME,N2)"""
    return (EMA(VOLUME, N1) - EMA(VOLUME, N2)) / EMA(VOLUME, N2)


def BIASVOL(VOLUME, N=6):  # 成交量乖离率
    """BIASVOL=(VOLUME-MA(VOLUME,N))/MA(VOLUME,N)*100，修复除零警告"""
    ma_volume = MA(VOLUME, N)  # 计算成交量的N日移动平均
    epsilon = 1e-10  # 处理浮点数精度误差（避免极小数被误判）
    # 过滤分母为0或NaN的情况：这些情况返回0，否则正常计算
    valid = ~np.isnan(ma_volume) & (np.abs(ma_volume) > epsilon)
    return np.where(valid, (VOLUME - ma_volume) / ma_volume * 100, 0.0)


def MACDVOL(VOLUME, N1=20, N2=40, N3=10):  # 成交量MACD
    """MACD=EMA(VOLUME,N1)-EMA(VOLUME,N2)
       SIGNAL=MA(MACD,N3)"""
    MACD = EMA(VOLUME, N1) - EMA(VOLUME, N2)
    return MACD, MA(MACD, N3)


def ROCVOL(VOLUME, N=80):  # 成交量ROC
    """ROCVOL=(VOLUME-REF(VOLUME,N))/REF(VOLUME,N)*100，修复除零警告"""
    ref_volume = REF(VOLUME, N)  # 获取N期前的成交量
    epsilon = 1e-10  # 处理浮点数精度误差（避免极小数被误判）
    # 过滤分母为0或NaN的情况：有效时计算，无效时返回0
    valid = ~np.isnan(ref_volume) & (np.abs(ref_volume) > epsilon)
    return np.where(valid, (VOLUME - ref_volume) / ref_volume * 100, 0.0)


# 表37:价量指标

def VWAP(CLOSE, HIGH, LOW, VOLUME, N=20):  # 成交量加权均价
    """Typical=(HIGH+LOW+CLOSE)/3
       MF=VOLUME*Typical
       VWAP=SUM(MF,N)/SUM(VOLUME,N)"""
    Typical = (HIGH + LOW + CLOSE) / 3
    MF = Typical * VOLUME
    return SUM(MF, N) / SUM(VOLUME, N)


def FI(CLOSE, VOLUME, N=13):  # 资金流量
    """FI=(CLOSE-REF(CLOSE,1))*VOLUME
       FIMA=EMA(FI,N)"""
    FI = (CLOSE - REF(CLOSE, 1)) * VOLUME
    return EMA(FI, N)

def CUM_PROD(S):
    return np.cumprod(S)
def NVI(CLOSE, VOLUME, N=144):  # 负成交量指标
    """NVI_INC=IF(VOLUME<REF(VOLUME,1),1+(CLOSE-REF(CLOSE,1))/CLOSE,1)
       NVI=CUM_PROD(NVI_INC)"""
    NVI_INC = IF(VOLUME < REF(VOLUME, 1), 1 + (CLOSE - REF(CLOSE, 1)) / CLOSE, 1)
    return CUM_PROD(NVI_INC)


def PVT(CLOSE, VOLUME, N1=13, N2=34):  # 价格成交量趋势
    """PVT=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*VOLUME
       PVT_MA1=MA(PVT,N1)
       PVT_MA2=MA(PVT,N2)"""
    PVT = (CLOSE - REF(CLOSE, 1)) / REF(CLOSE, 1) * VOLUME
    return MA(PVT, N1), MA(PVT, N2)


def RSIV(VOLUME, CLOSE, N=20):  # 成交量RSI
    """VOLUP=IF(CLOSE>REF(CLOSE,1),VOLUME,0)
       VOLDOWN=IF(CLOSE<REF(CLOSE,1),VOLUME,0)
       SUMUP=SUM(VOLUP,N)
       SUMDOWN=SUM(VOLDOWN,N)
       RSIV=100*SUMUP/(SUMUP+SUMDOWN)"""
    VOLUP = IF(CLOSE > REF(CLOSE, 1), VOLUME, 0)
    VOLDOWN = IF(CLOSE < REF(CLOSE, 1), VOLUME, 0)
    SUMUP = SUM(VOLUP, N)
    SUMDOWN = SUM(VOLDOWN, N)
    return 100 * SUMUP / (SUMUP + SUMDOWN)


def AMV(OPEN, CLOSE, VOLUME, N1=13, N2=34):  # 成交均价
    """AMOV=VOLUME*(OPEN+CLOSE)/2
       AMV1=SUM(AMOV,N1)/SUM(VOLUME,N1)
       AMV2=SUM(AMOV,N2)/SUM(VOLUME,N2)"""
    AMOV = VOLUME * (OPEN + CLOSE) / 2
    AMV1 = SUM(AMOV, N1) / SUM(VOLUME, N1)
    AMV2 = SUM(AMOV, N2) / SUM(VOLUME, N2)
    return AMV1, AMV2


def VRAMT(AMOUNT, CLOSE, N=40):  # 成交金额变异率
    """AV=IF(CLOSE>REF(CLOSE,1),AMOUNT,0)
       BV=IF(CLOSE<REF(CLOSE,1),AMOUNT,0)
       CVS=SUM(IF(CLOSE==REF(CLOSE,1),AMOUNT,0),N)
       VRAMT=(SUM(AV,N)+CVS/2)/(SUM(BV,N)+CVS/2)"""
    AV = IF(CLOSE > REF(CLOSE, 1), AMOUNT, 0)
    BV = IF(CLOSE < REF(CLOSE, 1), AMOUNT, 0)
    CVS = SUM(IF(CLOSE == REF(CLOSE, 1), AMOUNT, 0), N)
    return (SUM(AV, N) + CVS / 2) / (SUM(BV, N) + CVS / 2)


def WVAD(CLOSE, OPEN, HIGH, LOW, VOLUME, N=20):  # 威廉变异离散量指标
    """
    计算威廉变异离散量（WVAD），修复除零警告
    CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    OPEN: 开盘价序列
    HIGH: 最高价序列
    LOW: 最低价序列
    VOLUME: 成交量序列
    N: 滚动窗口大小（默认20）
    返回：WVAD序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    for s in [CLOSE, OPEN, HIGH, LOW, VOLUME]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            OPEN = pd.Series(OPEN)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            VOLUME = pd.Series(VOLUME)
            break  # 统一转换后退出检查
    
    # 2. 计算核心比率的分子和分母
    numerator = CLOSE - OPEN  # 分子：收盘价 - 开盘价
    denominator = HIGH - LOW  # 分母：最高价 - 最低价
    
    # 3. 处理除零问题：分母为0或极接近0时，比率设为NaN
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    valid = (denominator > epsilon) & ~np.isnan(denominator) & ~np.isnan(numerator)
    ratio = np.where(valid, numerator / denominator, np.nan)  # 安全计算比率
    
    # 4. 计算比率×成交量（无效值会保持NaN）
    ratio_volume = ratio * VOLUME
    
    # 5. 实现滚动SUM（N期内求和，而非全局求和）
    def rolling_sum(X, window):
        """滚动窗口求和，数据不足时返回NaN"""
        return pd.Series(X).rolling(window=window, min_periods=window).sum()
    
    # 6. 计算WVAD：N期内(比率×成交量)的滚动和
    WVAD = rolling_sum(ratio_volume, N)
    
    return WVAD


def OBV(CLOSE, VOLUME):  # 能量潮
    """OBV=IF(CLOSE>REF(CLOSE,1),VOLUME,IF(CLOSE<REF(CLOSE,1),-VOLUME,0))
       OBV=CUMSUM(OBV)"""
    return CUM_SUM(IF(CLOSE > REF(CLOSE, 1), VOLUME, IF(CLOSE < REF(CLOSE, 1), -VOLUME, 0)))


def CMF(CLOSE, HIGH, LOW, VOLUME, N=20):  # 修复版蔡金货币流量（CMF）
    """
    计算蔡金货币流量指标（CMF），修复除零警告
    :param CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    :param HIGH: 最高价序列
    :param LOW: 最低价序列
    :param VOLUME: 成交量序列
    :param N: 滚动计算周期（默认20）
    :return: CMF指标序列
    """
    # 1. 标准化输入为pandas.Series，确保滚动计算索引对齐
    for s in [CLOSE, HIGH, LOW, VOLUME]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            VOLUME = pd.Series(VOLUME)
            break
    
    # 2. 安全计算CLV：过滤分母为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差（如0.0000000001视为0）
    clv_numerator = 2 * CLOSE - LOW - HIGH  # CLV分子
    clv_denominator = HIGH - LOW            # CLV分母（当天价格波动范围）
    
    # 有效性条件：分母非0+非NaN + 分子非NaN（避免无效运算）
    valid_clv = (
        (clv_denominator > epsilon) & ~np.isnan(clv_denominator) &
        ~np.isnan(clv_numerator)
    )
    
    # 特殊处理：当天无波动时（HIGH==LOW），CLV理论为0（收盘价=高低价）
    CLV = np.where(valid_clv, clv_numerator / clv_denominator, 0.0)
    CLV = pd.Series(CLV, index=CLOSE.index)  # 保持索引一致
    
    # 3. 实现正确的滚动SUM（N期内求和，而非全局求和）
    def rolling_sum(X, window):
        """N期滚动求和：仅窗口内有window个有效数据时返回值，否则为NaN"""
        return X.rolling(window=window, min_periods=window).sum()
    
    # 4. 计算CMF的分子和分母（滚动求和）
    cmf_numerator = rolling_sum(CLV * VOLUME, N)  # 分子：CLV×成交量的N期滚动和
    cmf_denominator = rolling_sum(VOLUME, N)      # 分母：成交量的N期滚动和
    
    # 5. 处理CMF的二次除零（避免成交量滚动和为0的极端情况）
    valid_cmf = (
        (cmf_denominator > epsilon) & ~np.isnan(cmf_denominator) &
        ~np.isnan(cmf_numerator)
    )
    CMF = np.where(valid_cmf, cmf_numerator / cmf_denominator, np.nan)
    
    return pd.Series(CMF, index=CLOSE.index)


def PVI(CLOSE, VOLUME, N=40):  # 正成交量指标
    """PVI_INC=IF(VOLUME>REF(VOLUME,1),1+(CLOSE-REF(CLOSE,1))/CLOSE,1)
       PVI=CUM_PROD(PVI_INC)"""
    PVI_INC = IF(VOLUME > REF(VOLUME, 1), 1 + (CLOSE - REF(CLOSE, 1)) / CLOSE, 1)
    return CUM_PROD(PVI_INC)


def TMF(CLOSE, HIGH, LOW, VOLUME, N=20):  # 真实货币流量
    """TRH=MAX(HIGH,REF(CLOSE,1))
       TRL=MIN(LOW,REF(CLOSE,1))
       AD=IF(CLOSE>REF(CLOSE,1),CLOSE-TRL,IF(CLOSE<REF(CLOSE,1),CLOSE-TRH,0))
       TMF=MA(AD,N)"""
    TRH = MAX(HIGH, REF(CLOSE, 1))
    TRL = MIN(LOW, REF(CLOSE, 1))
    AD = IF(CLOSE > REF(CLOSE, 1), CLOSE - TRL, IF(CLOSE < REF(CLOSE, 1), CLOSE - TRH, 0))
    return MA(AD, N)


def MFI(CLOSE, HIGH, LOW, VOLUME, N=14):  # 资金流量指标
    """
    计算资金流量指标（MFI），修复除零警告
    CLOSE: 收盘价序列（pandas.Series或numpy.ndarray）
    HIGH: 最高价序列
    LOW: 最低价序列
    VOLUME: 成交量序列
    N: 滚动窗口大小（默认14）
    返回：MFI序列
    """
    # 1. 标准化输入为pandas.Series，确保索引对齐
    for s in [CLOSE, HIGH, LOW, VOLUME]:
        if not isinstance(s, pd.Series):
            CLOSE = pd.Series(CLOSE)
            HIGH = pd.Series(HIGH)
            LOW = pd.Series(LOW)
            VOLUME = pd.Series(VOLUME)
            break  # 统一转换后退出检查
    
    # 2. 计算典型价格（TYPICAL）和资金流量（MF）
    TYPICAL = (HIGH + LOW + CLOSE) / 3  # 典型价格
    MF = TYPICAL * VOLUME  # 资金流量
    
    # 3. 计算前一期典型价格（替代REF(TYPICAL, 1)）
    ref_typical = TYPICAL.shift(1)  # 前1期典型价格
    
    # 4. 计算正向/负向资金流量（MF_POS/MF_NEG）
    # 正向：当前典型价格≥前一期时，取MF，否则取0
    mf_pos = np.where(TYPICAL >= ref_typical, MF, 0)
    # 负向：当前典型价格≤前一期时，取MF，否则取0
    mf_neg = np.where(TYPICAL <= ref_typical, MF, 0)
    
    # 5. 滚动求和（N期内的累计，替代SUM函数）
    def rolling_sum(X, window):
        """滚动窗口求和，数据不足时返回NaN"""
        return pd.Series(X).rolling(window=window, min_periods=window).sum()
    
    MF_POS = rolling_sum(mf_pos, N)  # N期正向资金总和
    MF_NEG = rolling_sum(mf_neg, N)  # N期负向资金总和
    
    # 6. 处理除零问题，计算MFI
    epsilon = 1e-10  # 处理浮点数精度误差
    # 情况1：MF_NEG为0或极接近0（此时MFI理论上为100）
    is_neg_zero = (MF_NEG < epsilon) & ~np.isnan(MF_NEG) & ~np.isnan(MF_POS)
    # 情况2：MF_POS和MF_NEG均为NaN（数据不足）
    is_nan = np.isnan(MF_POS) | np.isnan(MF_NEG)
    
    # 安全计算比率：仅在有效时执行除法
    ratio = np.where(
        ~is_neg_zero & ~is_nan,  # 排除无效情况
        MF_POS / MF_NEG,
        np.nan  # 无效时先设为NaN
    )
    
    # 计算最终MFI：特殊情况单独处理
    MFI = np.where(
        is_neg_zero,
        100.0,  # MF_NEG=0时，MFI=100
        np.where(
            is_nan,
            np.nan,  # 数据不足时为NaN
            100 - 100 / (1 + ratio)  # 正常情况计算公式
        )
    )
    
    return pd.Series(MFI, index=CLOSE.index)


def ADOSC(AD, N1=10, N2=30):  # 震荡升降指标
    """ADOSC=EMA(AD,N1)-EMA(AD,N2)"""
    return EMA(AD, N1) - EMA(AD, N2)


def VAO(CLOSE, HIGH, LOW, VOLUME, N1=10, N2=30):  # 价量变异率
    """WEIGHTED_VOLUME=VOLUME*(CLOSE-(HIGH+LOW)/2)
       VAO=CUMSUM(WEIGHTED_VOLUME)
       VAO_MA1=MA(VAO,N1)
       VAO_MA2=MA(VAO,N2)"""
    WEIGHTED_VOLUME = VOLUME * (CLOSE - (HIGH + LOW) / 2)
    VAO = CUM_SUM(WEIGHTED_VOLUME)
    return MA(VAO, N1), MA(VAO, N2)


def VR(VOLUME, CLOSE, N=40):  # 成交量变异率
    """AV=SUM(IF(CLOSE>REF(CLOSE,1),VOLUME,0),N)
       BV=SUM(IF(CLOSE<REF(CLOSE,1),VOLUME,0),N)
       CVS=SUM(IF(CLOSE==REF(CLOSE,1),VOLUME,0),N)
       VR=(AV+CVS/2)/(BV+CVS/2)"""
    AV = SUM(IF(CLOSE > REF(CLOSE, 1), VOLUME, 0), N)
    BV = SUM(IF(CLOSE < REF(CLOSE, 1), VOLUME, 0), N)
    CVS = SUM(IF(CLOSE == REF(CLOSE, 1), VOLUME, 0), N)
    return (AV + CVS / 2) / (BV + CVS / 2)


def KO(CLOSE, HIGH, LOW, VOLUME, N1=34, N2=55):  # 知识能量指标
    """TYPICAL=(HIGH+LOW+CLOSE)/3
       VOLUME=IF(TYPICAL>REF(TYPICAL,1),VOLUME,-VOLUME)
       VOLUME_EMA1=EMA(VOLUME,N1)
       VOLUME_EMA2=EMA(VOLUME,N2)
       KO=VOLUME_EMA1-VOLUME_EMA2"""
    TYPICAL = (HIGH + LOW + CLOSE) / 3
    VOLUME_SIGNED = IF(TYPICAL > REF(TYPICAL, 1), VOLUME, -VOLUME)
    VOLUME_EMA1 = EMA(VOLUME_SIGNED, N1)
    VOLUME_EMA2 = EMA(VOLUME_SIGNED, N2)
    return VOLUME_EMA1 - VOLUME_EMA2


def EMV(CLOSE, HIGH, LOW, VOLUME, N=20):  # 简易波动指标
    """EMV=((HIGH+LOW)/2-REF((HIGH+LOW)/2,1))/VOLUME*10000
       EMVMA=MA(EMV,N)，修复除零警告"""
    # 计算分子部分：((HIGH + LOW)/2 - 前一期的(HIGH + LOW)/2) * 10000
    price_mid = (HIGH + LOW) / 2
    price_mid_diff = price_mid - REF(price_mid, 1)
    numerator = price_mid_diff * 10000  # 先乘10000，避免后续计算顺序问题

    # 处理成交量为0或极接近0的情况
    epsilon = 1e-10  # 处理浮点数精度误差
    valid = ~np.isnan(VOLUME) & (np.abs(VOLUME) > epsilon)  # 成交量有效（非0且非NaN）

    # 有效时正常计算EMV，无效时设为0（成交量为0时波动视为0）
    EMV = np.where(valid, numerator / VOLUME, 0.0)

    # 计算EMV的N日移动平均
    return MA(EMV, N)


# # 表38:大盘指标
#
# def STIX(UP_STOCK, DOWN_STOCK, N=40):  # 短期指数
#     """STIX=EMA(UP_STOCK/(UP_STOCK+DOWN_STOCK)*100,N)"""
#     TOTAL = UP_STOCK + DOWN_STOCK
#     return EMA(UP_STOCK / TOTAL * 100, N)
#
#
# def MIO(UP_STOCK, DOWN_STOCK, N1=40, N2=80):  # 市场情绪指标
#     """MIO=EMA(UP_STOCK-DOWN_STOCK,N1)-EMA(UP_STOCK-DOWN_STOCK,N2)"""
#     AD = UP_STOCK - DOWN_STOCK
#     return EMA(AD, N1) - EMA(AD, N2)
#
#
# def ADIO(UP_STOCK, DOWN_STOCK, N=40):  # 涨跌差指标
#     """ADIO=MA(UP_STOCK-DOWN_STOCK,N)"""
#     return MA(UP_STOCK - DOWN_STOCK, N)
#
#
# def MCL(UP_STOCK, DOWN_STOCK, N1=40, N2=80):  # 麦克连指标
#     """MCL=EMA(UP_STOCK-DOWN_STOCK,N1)-EMA(UP_STOCK-DOWN_STOCK,N2)"""
#     AD = UP_STOCK - DOWN_STOCK
#     return EMA(AD, N1) - EMA(AD, N2)
#
#
# def ADIPO(UP_STOCK, DOWN_STOCK, N=40):  # 涨跌比率
#     """ADIPO=MA((UP_STOCK-DOWN_STOCK)/(UP_STOCK+DOWN_STOCK)*100,N)"""
#     TOTAL = UP_STOCK + DOWN_STOCK
#     return MA((UP_STOCK - DOWN_STOCK) / TOTAL * 100, N)
#
#
# def ADVR(UP_VOLUME, DOWN_VOLUME, N=10, M=10):  # 涨跌量比
#     """ADVOLUME=MA(UP_VOLUME,N)
#        DEVOLUME=MA(DOWN_VOLUME,N)
#        ADVR=MA(ADVOLUME/DEVOLUME,M)"""
#     ADVOLUME = MA(UP_VOLUME, N)
#     DEVOLUME = MA(DOWN_VOLUME, N)
#     return MA(ADVOLUME / DEVOLUME, M)
#
#
# def ADVPO(UP_VOLUME, DOWN_VOLUME, N=20):  # 成交量差
#     """ADVPO=MA((UP_VOLUME-DOWN_VOLUME)/(UP_VOLUME+DOWN_VOLUME)*100,N)"""
#     TOTAL_VOLUME = UP_VOLUME + DOWN_VOLUME
#     return MA((UP_VOLUME - DOWN_VOLUME) / TOTAL_VOLUME * 100, N)
#
#
# def ADR(UP_STOCK, DOWN_STOCK, N=40):  # 涨跌比率
#     """ADR=MA(UP_STOCK/DOWN_STOCK,N)"""
#     return MA(UP_STOCK / DOWN_STOCK, N)
#
#
# def CVI(UP_VOLUME, DOWN_VOLUME, N=20):  # 累积成交量差
#     """CVI=CUMSUM(UP_VOLUME-DOWN_VOLUME)
#        CVIMA=MA(CVI,N)"""
#     CVI = CUM_SUM(UP_VOLUME - DOWN_VOLUME)
#     return MA(CVI, N)
#
#
# def BT(UP_STOCK, DOWN_STOCK, N=20):  # 布林带
#     """BT=MA(UP_STOCK,N)/MA(UP_STOCK+DOWN_STOCK,N)*100"""
#     TOTAL = UP_STOCK + DOWN_STOCK
#     return MA(UP_STOCK, N) / MA(TOTAL, N) * 100
#
#
# def TRIN(UP_STOCK, DOWN_STOCK, UP_VOLUME, DOWN_VOLUME, N=20):  # 三一指标
#     """TRIN=MA((UP_STOCK/DOWN_STOCK)/(UP_VOLUME/DOWN_VOLUME),N)"""
#     ratio_stock = UP_STOCK / DOWN_STOCK
#     ratio_volume = UP_VOLUME / DOWN_VOLUME
#     return MA(ratio_stock / ratio_volume, N)