# 015_indicator_scanner

量化指标扫描与模拟盘跟踪系统。在沪深300成分股中，用全量技术指标扫描找到最稳健的普适性指标，再选出该指标下表现最好的股票，每日进行模拟盘跟踪，通过 WxPusher 推送日报。

## 核心思路

> **先找最稳健的指标，再用它选股** — 避免"每只股票挑不同指标"带来的双重过拟合

### 四阶段流水线

```
Phase 1 (扫描):  300只股票 × 97个指标 × 2年回测 → 找到普适性最强的指标
Phase 2 (选股):  300只股票 × 1个最佳指标 → 超额收益 Top 10
Phase 3 (验证):  10只股票 × 近3个月 → 确认信号未失效
Phase 4 (模拟):  每交易日21:00 → 执行订单 + 生成信号 + 推送日报
```

### 综合评分公式

```
score = mean_excess_return × win_rate - 0.5 × std_excess_return
```

- `mean_excess_return`：指标在所有股票上的平均超额收益 — 越高越好
- `win_rate`：指标跑赢基准的股票比例 — 衡量普适性
- `std_excess_return`：超额收益的标准差 — 惩罚不稳定指标

## 项目结构

```
015_indicator_scanner/
├── run_scanner.py              # 唯一入口（cron 调用）
├── config/
│   ├── __init__.py
│   └── scanner_config.py       # 全局配置（路径/资金/WxPusher/参数）
├── core/                       # 回测引擎（从 002_self_backtest_reborn 复制+裁剪）
│   ├── engine.py               # BacktestEngine / BacktestConfig
│   ├── data_loader.py          # DataLoader（CSV → DataFrame）
│   ├── signal_engine.py        # SignalEngine / BaseSignal
│   ├── risk_manager.py         # RiskManager（涨跌停/止盈止损）
│   ├── equity_curve.py         # EquityCurveCalculator
│   ├── metrics.py              # MetricsCalculator
│   ├── reporter.py             # Reporter
│   ├── log_utils.py            # 日志工具
│   ├── scanner.py              # [新建] Phase 1-3 扫描编排器
│   ├── simulator.py            # [新建] Phase 4 每日模拟盘
│   ├── state_manager.py        # [新建] JSON 状态持久化
│   ├── notification.py         # [新建] WxPusher 推送
│   └── hs300_utils.py          # [新建] 沪深300成分股/交易日判断
├── signals/                    # 信号模块（从 002_self_backtest_reborn 复制+裁剪）
│   ├── gf.py                   # GFSignal（97个指标信号方法）
│   └── gf_factors.py           # 143+ 指标计算函数（numpy）
├── output/
│   ├── state.json              # 状态文件（运行时创建）
│   ├── scans/                  # Phase 1-3 产物
│   └── daily/                  # Phase 4 日报 CSV
└── logs/                       # 运行日志
```

## 快速上手

### 1. 环境准备

```bash
source activate zhulei
cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner
```

### 2. 小规模测试

```bash
# 用 5 个指标 × 10 只股票快速验证全流程
python run_scanner.py --test-mode --force
```

### 3. 全量扫描

```bash
# 完整扫描（97个指标 × 300只股票，约15-30分钟）
python run_scanner.py --force
```

### 4. 每日模拟盘

```bash
# 仅执行 Phase 4（通常由 cron 自动调用）
python run_scanner.py --phase 4

# Dry-run 模式（输出操作但不修改状态）
python run_scanner.py --phase 4 --dry-run
```

### 5. 单独运行各阶段

```bash
python run_scanner.py --phase 1    # 仅扫描指标
python run_scanner.py --phase 2    # 仅选股
python run_scanner.py --phase 3    # 仅验证
```

## Cron 配置

```bash
# 每日模拟盘（交易日 21:00，周一至周五）
0 21 * * 1-5 cd /path/to/015_indicator_scanner && \
  /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py >> logs/daily_$(date +\%Y\%m\%d).log 2>&1

# 每季度扫描（3/6/9/12月1日凌晨2:00）
0 2 1 3,6,9,12 * cd /path/to/015_indicator_scanner && \
  /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py --force >> logs/quarterly_$(date +\%Y\%m\%d).log 2>&1
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--phase auto` | 自动判断执行阶段 | `auto` |
| `--phase 1/2/3/4` | 强制运行指定阶段 | — |
| `--force` | 强制重新扫描（忽略状态） | — |
| `--dry-run` | Phase 4 dry-run（不修改状态） | — |
| `--test-mode` | 小规模测试（5指标×10股票） | — |

## 状态文件 (state.json)

```json
{
  "version": 1,
  "current_phase": "idle|scanning|selecting|verifying|running",
  "best_indicator": "KDJ",
  "top_10_stocks": ["600000", "600009", ...],
  "verify_passed": true,
  "portfolio": {
    "cash": 950000.0,
    "initial_capital": 1000000.0,
    "positions": {},
    "pending_orders": {}
  },
  "trade_log": [...],
  "last_scan_date": "2026-06-01"
}
```

## 数据来源

- **股票日线**：`../data/input/{code}.csv`（5207只A股，交易日晚间更新）
- **沪深300成分股**：通过 baostock API 获取（7天缓存）
- **指数日线**：`../data/input/index_k_data/hushen300.csv`
- **交易日判断**：参考股票 CSV 最后一行日期是否等于今天

## 关键技术细节

- **消除 look-ahead bias**：信号在第二天开盘执行，与回测逻辑完全一致
- **原子写入**：state.json 先写 `.tmp` 再 rename，防止并发损坏
- **轻量回测**：Phase 1 不写文件、不画图、不打印日志，最大化吞吐
- **扁平并行**：所有 (indicator, stock) 对一次性提交到线程池（32线程）
- **数据预加载**：300 只股票的 DataFrame 一次读入内存（~18 MB）
- **非交易日静默**：Cron 每日触发，内部判断交易日，非交易日直接退出

## 与 002_self_backtest_reborn 的关系

本项目的核心回测模块（`core/` 和 `signals/`）从 `002_self_backtest_reborn` 复制而来，裁剪了不需要的部分：

| 模块 | 处理 |
|------|------|
| `core/comparator.py` | 不复制（本项目的 scanner.py 替代） |
| `core/optimizer.py` | 不复制 |
| `core/risk_manager.py` | 移除 `apply_trailing_stop()` |
| `core/engine.py` | 移除 `trailing_stop_pct` / `trailing_profit_pct` |
| `signals/gf.py` | 移除 `ComboGFSignal` 类 |

两个项目独立维护，互不影响。
