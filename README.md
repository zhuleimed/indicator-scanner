# 📈 015_indicator_scanner — 指标扫描与模拟盘系统

> 基于 97 个技术指标对沪深300成分股进行全量扫描，自动选出最稳健的普适性指标和最优股票，每日模拟盘跟踪并推送日报。

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

---

## ⚡ 快速开始

```bash
# 1. 激活环境
source activate zhulei

# 2. 进到项目目录
cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner

# 3. 小规模测试（5指标 × 10股票，约15秒）
python run_scanner.py --test-mode --force

# 4. 全量扫描（97指标 × 300股票，约15-30分钟）
python run_scanner.py --force

# 5. 查看状态
cat output/state.json | python -m json.tool
```

## 🔄 四阶段流水线

```
Phase 1  全指标扫描      300股 × 97指标 × 2年 → 最佳指标
    ↓
Phase 2  选股            300股 × 1最佳指标 → Top 10 股票
    ↓
Phase 3  验证            近3个月验证 → 通过/失败
    ↓
Phase 4  模拟盘          每日交易执行 + 信号生成 + 微信推送
```

### 综合评分公式

```
score = mean_excess_return × win_rate - 0.5 × std_excess_return
```

选择在多数股票上都能持续小幅跑赢基准的**稳健型指标**，而非少数股票上暴赚的不稳定指标。

## 📋 常用命令

| 命令 | 说明 |
|------|------|
| `python run_scanner.py` | 自动判断阶段（日常使用） |
| `python run_scanner.py --force` | 强制重新扫描 |
| `python run_scanner.py --phase 4` | 仅执行模拟盘 |
| `python run_scanner.py --dry-run` | 模拟盘 dry-run（不修改状态） |
| `python run_scanner.py --test-mode` | 小规模测试 |

## ⏰ Cron 定时任务

### 配置方法

```bash
# 编辑 crontab
crontab -e

# 添加以下两行：
# 每日模拟盘（交易日 21:00，非交易日自动跳过）
0 21 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner && /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py >> logs/daily_$(date +\%Y\%m\%d).log 2>&1

# 每季度重扫描（3/6/9/12月1日凌晨2:00）
0 2 1 3,6,9,12 * cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner && /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py --force >> logs/quarterly_$(date +\%Y\%m\%d).log 2>&1
```

> **重要**：首次运行前，先手动执行 `python run_scanner.py --force` 完成全量扫描。

## 📊 数据来源

| 阶段 | 数据源 |
|------|--------|
| Phase 1-3（扫描/选股/验证） | 本地 CSV 文件 |
| Phase 4（每日模拟盘） | **baostock API 实时拉取** |

```
output/
├── state.json              # 状态文件（扫描结果/持仓/交易记录）
├── hs300_stocks.csv        # 成分股缓存（7天有效）
├── scans/                  # Phase 1-3 产物
│   ├── scan_*.json         # 扫描结果
│   └── selection_*.json    # 选股结果
└── daily/                  # Phase 4 日报
    └── daily_*.csv
```

## 🛠 配置

所有参数集中在 `config/scanner_config.py`：

- **数据路径**：`DATA_DIR`、`INDEX_DIR`
- **资金参数**：`INITIAL_CAPITAL`（默认100万）、`SLIPPAGE`（0.3%）
- **并行**：`MAX_WORKERS`（默认32）
- **选股数**：`TOP_N_STOCKS`（默认10）
- **WxPusher**：`WXPUSHER_TOKEN`、`WXPUSHER_UIDS`

---

> **Happy Scanning! 🔍**
