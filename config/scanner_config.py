"""
扫描器全局配置 — 015_indicator_scanner

所有路径、参数、常量集中管理。运行前请确认以下配置正确：
  - WxPusher token / uids / topic_ids
  - 数据目录路径
  - 资金与交易成本参数
"""

import os

# ============================================================================
# 路径配置
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(PROJECT_DIR)       # .../code/
QUANT_DIR = os.path.dirname(CODE_DIR)          # .../quant/

# 股票日线数据目录: quant/data/input/
DATA_DIR = os.path.join(QUANT_DIR, 'data', 'input')

# 指数日线数据目录
INDEX_DIR = os.path.abspath(os.path.join(DATA_DIR, 'index_k_data'))

# 沪深300指数文件
HS300_INDEX_FILE = os.path.join(INDEX_DIR, 'hushen300.csv')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
SCANS_DIR = os.path.join(OUTPUT_DIR, 'scans')
DAILY_DIR = os.path.join(OUTPUT_DIR, 'daily')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

# 状态文件
STATE_FILE = os.path.join(OUTPUT_DIR, 'state.json')

# ============================================================================
# 扫描参数
# ============================================================================

# 扫描回测期（年）
SCAN_YEARS = 2

# 验证期（月）
VERIFY_MONTHS = 3

# 重扫间隔（天）
RESCAN_DAYS = 90

# 选股数量
TOP_N_STOCKS = 10

# 验证通过阈值（至少 N/10 只股票跑赢基准）
VERIFY_PASS_THRESHOLD = 6

# ============================================================================
# 并行计算
# ============================================================================

# 最大并行工作进程数（ProcessPoolExecutor）
MAX_WORKERS = 32

# ============================================================================
# 综合评分公式参数
#   score = mean_excess_return × WIN_RATE_WEIGHT - std_excess_return × STD_PENALTY
# ============================================================================

# 标准差惩罚系数（越大越偏好稳定指标）
STD_PENALTY = 0.5

# ============================================================================
# 资金与交易成本
# ============================================================================

# 模拟盘初始资金（元）
INITIAL_CAPITAL = 1_000_000.0

# 每只股票初始资金分配（初始等权，随时间漂移）
INITIAL_MONEY_PER_STOCK = 100_000.0

# 滑点（0.003 = 0.3%）
SLIPPAGE = 0.003

# 佣金比例（0.0005 = 万分之五，最低5元）
COMMISSION_RATE = 0.0005

# 印花税比例（0.001 = 千分之一，卖出时收取）
TAX_RATE = 0.001

# 仓位比例（0.95 = 95%）
POSITION_PCT = 0.95

# ============================================================================
# 风控参数（与 002_self_backtest_reborn 保持一致）
# ============================================================================

# 止损比例（5%）
STOP_LOSS_PCT = 0.05

# 止盈触发比例（20%）
STOP_PROFIT_PCT = 0.20

# 回落止盈比例（3%）
DRAWDOWN_PCT = 0.03

# ============================================================================
# 基准与无风险利率
# ============================================================================

# 基准指数代码
BENCHMARK_CODE = 'sh.000300'

# 无风险利率（2.7%）
RISK_FREE_RATE = 0.027

# ============================================================================
# WxPusher 微信推送配置
# ============================================================================

WXPUSHER_TOKEN = 'AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX'
WXPUSHER_UIDS = ['<uids>']
WXPUSHER_TOPIC_IDS = ['39277']

# ============================================================================
# 数据过滤
# ============================================================================

# 次新股过滤：上市不足 MIN_TRADING_DAYS 个交易日的股票跳过
MIN_TRADING_DAYS = 240

# 用于判断交易日的参考股票代码（只需一只数据完整的股票）
REFERENCE_STOCK = '000001'


def ensure_dirs():
    """确保所有输出目录存在。"""
    for d in [OUTPUT_DIR, SCANS_DIR, DAILY_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
