# 📈 指标扫描与模拟盘系统

> **015_indicator_scanner** — 基于 97 个技术指标对沪深300成分股进行全自动扫描，选出最稳健指标和最优股票，每日模拟盘跟踪并通过 WxPusher 微信推送日报

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-indicator--scanner-green)](https://github.com/zhuleimed/indicator-scanner)

---

## 📑 目录

- [一分钟快速上手](#-一分钟快速上手)
- [核心理念](#-核心理念)
- [项目结构](#-项目结构)
- [四阶段流水线](#-四阶段流水线)
- [命令详解（run_scanner.py）](#-命令详解run_scannerpy)
- [Cron 定时与运维](#-cron-定时与运维)
- [输出结果解读](#-输出结果解读)
- [配置参数说明](#-配置参数说明)
- [常见问题](#-常见问题)

---

## ⚡ 一分钟快速上手

```bash
# 1. 激活环境 + 进到项目目录
source activate zhulei
cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner

# 2. 小规模测试（5个指标 × 10只股票，约5秒）
python run_scanner.py --test-mode --force

# 3. 全量扫描（97个指标 × 300只股票，约5-10分钟）
python run_scanner.py --force

# 4. 看扫描结果
cat output/state.json | python -m json.tool | head -30

# 5. 手动执行模拟盘（dry-run 不修改状态）
python run_scanner.py --phase 4 --dry-run
```

---

## 💡 核心理念

> **先找最稳健的指标，再用它选股** — 避免"每只股票挑不同指标"带来的双重过拟合

### 基准逻辑

超额收益 = **策略收益 − 该股票自身的 buy-and-hold 收益**

- 股票涨了 50%，策略赚了 60% → 超额 +10%（跑赢持股不动）
- 股票跌了 30%，策略持有现金 0% → 超额 +30%（避开了下跌）
- 股票涨了 100%，策略赚了 80% → 超额 −20%（择时交易错过了部分涨幅）

### 综合评分公式

```
score = mean_excess_return × win_rate − 0.5 × std_excess_return
```

| 维度 | 含义 | 作用 |
|------|------|------|
| `mean_excess_return` | 所有股票上的平均超额收益 | 衡量"能赚多少" |
| `win_rate` | 超额收益>0的股票占比 | 衡量"多大概率赚" |
| `std_excess_return` | 不同股票间超额收益的波动 | 惩罚"靠少数股票拉高均值" |

选出的是**在多数股票上都能稳定小幅跑赢**的稳健型指标，而非少数股票上暴赚的不稳定指标。

---

## 🏗 项目结构

```
015_indicator_scanner/
│
├── run_scanner.py              ← 🚀 唯一入口（所有操作通过它）
├── config/
│   └── scanner_config.py       ← 全局配置（路径/资金/WxPusher/参数）
│
├── core/                       ← ⚙️ 核心模块
│   ├── engine.py               ← 回测引擎（5步流水线）
│   ├── data_loader.py          ← 数据加载与清洗
│   ├── signal_engine.py        ← 信号合成引擎
│   ├── risk_manager.py         ← 风控（止盈止损、涨跌停）
│   ├── equity_curve.py         ← 资金曲线计算
│   ├── metrics.py              ← 绩效指标计算
│   ├── reporter.py             ← 结果输出与图表
│   ├── log_utils.py            ← 日志工具
│   ├── scanner.py              ← ⭐ Phase 1-3 扫描编排器（多进程并行）
│   ├── simulator.py            ← ⭐ Phase 4 每日模拟盘引擎
│   ├── state_manager.py        ← ⭐ 状态持久化（原子写入 JSON）
│   ├── notification.py         ← ⭐ WxPusher 微信推送
│   └── hs300_utils.py          ← ⭐ 沪深300成分股/交易日判断
│
├── signals/                    ← 📦 97个技术指标
│   ├── gf.py                   ← 信号策略类（97个 _signal_XXX 方法）
│   └── gf_factors.py           ← 指标计算函数库（143+ 个函数）
│
├── output/                     ← 📊 运行时产物
│   ├── state.json              ← 状态文件（扫描结果/持仓/交易记录）
│   ├── hs300_stocks.csv        ← 成分股缓存（7天有效）
│   ├── scans/                  ← Phase 1-3 结果
│   └── daily/                  ← Phase 4 日报
│
└── logs/                       ← 📜 运行日志（按天分文件）
    ├── daily_20260601.log
    └── quarterly_20260601.log
```

---

## 🔄 四阶段流水线

```
┌─ Phase 1: 扫描 ────────────────────────────────────┐
│ 97个指标 × 300只股票 × 2年回测 = ~29,000次轻量回测  │
│ 32进程并行，按综合评分排序                            │
│ 输出: 最佳指标名称                                   │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ Phase 2: 选股 ────────────────────────────────────┐
│ 用 Phase 1 选出的指标对 300 只股票逐一回测           │
│ 按超额收益降序排列                                   │
│ 输出: Top 10 股票列表                                │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ Phase 3: 验证 ────────────────────────────────────┐
│ Top 10 股票 × 最佳指标 × 近3个月                     │
│ 检查: ≥6/10 仍跑赢自身 buy-and-hold？               │
│ 通过 → 进入模拟盘   失败 → 重新扫描                   │
└────────────────────┬──────────────────────────────┘
                     ▼
┌─ Phase 4: 模拟盘（每个交易日 21:00）─────────────────┐
│ 1. 从 baostock API 拉取最新日线数据                  │
│ 2. 执行昨日待处理订单（今日开盘价成交）                │
│ 3. 生成明日信号（基于今日收盘数据）                    │
│ 4. 更新持仓/现金/待处理订单                           │
│ 5. WxPusher 推送日报到微信                           │
└────────────────────────────────────────────────────┘
```

---

## 📖 命令详解（`run_scanner.py`）

### 基本用法

```bash
# 最常用：自动判断执行什么（cron 也用这个）
python run_scanner.py

# 强制全量扫描（Phase 1→2→3，忽略当前状态）
python run_scanner.py --force

# 小规模测试（5指标×10股票，验证代码是否正常）
python run_scanner.py --test-mode --force
```

### 按阶段单独执行

```bash
# 仅执行 Phase 1（全指标扫描）
python run_scanner.py --phase 1

# 仅执行 Phase 2（选股，需 Phase 1 已完成）
python run_scanner.py --phase 2

# 仅执行 Phase 3（验证，需 Phase 1+2 已完成）
python run_scanner.py --phase 3

# 仅执行 Phase 4（模拟盘，dry-run 不修改状态）
python run_scanner.py --phase 4 --dry-run

# 仅执行 Phase 4（真实执行，会修改持仓和状态）
python run_scanner.py --phase 4
```

### --phase auto 决策逻辑

```
加载 state.json
  ├── needs_rescan() 或 phase=="idle" → 执行 Phase 1→2→3
  ├── phase=="running" 且是交易日   → 执行 Phase 4
  └── phase=="running" 且非交易日   → 退出
```

### 参数列表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--phase` | 枚举 | 执行阶段：`auto` / `1` / `2` / `3` / `4` | `auto` |
| `--force` | 标志 | 强制重新扫描（忽略当前状态和距上次扫描天数） | — |
| `--dry-run` | 标志 | Phase 4 仅输出操作不修改状态 | — |
| `--test-mode` | 标志 | 小规模测试（5指标×10股票） | — |

### 运行过程输出

**Phase 1 — 全指标扫描：**
```
============================================================
[Phase 1] 全指标扫描
  指标数: 97
  股票数: 300
  回测期: 2024-05-31 → 2026-05-31
  并行:   32 进程 (ProcessPoolExecutor)
============================================================

[1/2] 预检查有效股票…
  有效股票: 298/300

[2/2] 并行扫描 97 个指标…
  进度: 50/97 指标 (52%)  ETA: 45s
  完成! 耗时: 156s (2.6min)

计算综合评分…

──────────────────────────────────────────────────
  🏆 最佳指标 Top 10（综合评分）
──────────────────────────────────────────────────
  # 1 ADXR          score=-0.5345  mean_ex=-30.427%  win=29%  n=298
  # 2 KST           score=-0.6411  mean_ex=-45.617%  win=21%  n=298
  ...
```

**Phase 2 — 选股：**
```
============================================================
[Phase 2] 选股 — 指标: ADXR
  股票数: 298 (有效)
  并行:   32 进程
  基准:   每只股票自身的 buy-and-hold 收益率
============================================================

  🏆 Top 10 股票:
  # 1 688169  超额收益: 53.13%
  # 2 300979  超额收益: 46.43%
  ...
```

**Phase 3 — 验证：**
```
============================================================
[Phase 3] 验证 — 指标: ADXR
  股票数: 10
  验证期: 2026-02-27 → 2026-05-31
============================================================
  ✓ 688169  超额收益: 26.08%
  ✗ 002466  超额收益: -13.95%
  ...

  结果: 9/10 跑赢基准
  ✅ 验证通过！进入模拟盘阶段
```

**Phase 4 — 模拟盘日报（微信推送格式）：**
```
📊 指标扫描 · 模拟盘日报
日期: 2026-06-02
指标: ADXR

── 当日操作 ──
🟢 买入 688169: 500股 @ 45.32  成本=22660.00
🔴 卖出 300832: 300股 @ 78.15  盈亏=+3420.00  (signal_sell)

── 持仓摘要 ──
  688169: 500股  成本=45.32  现价=45.80  (+1.06%)
  600309: 200股  成本=32.10  现价=31.50  (-1.87%)

── 账户摘要 ──
总资产: 1,035,420.00
现金:   568,000.00
累计收益: +3.54%

── 明日信号 ──
🟢 601136: buy
🔴 002466: sell
```

---

## ⏰ Cron 定时与运维

### 当前配置

```bash
# 查看 crontab
crontab -l

# 活跃的两条规则：
# 每日模拟盘（交易日 21:00，周一至周五）
0 21 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner && /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py >> logs/daily_$(date +\%Y\%m\%d).log 2>&1

# 每季度重扫描（3/6/9/12月1日凌晨2:00）
0 2 1 3,6,9,12 * cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner && /home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py --force >> logs/quarterly_$(date +\%Y\%m\%d).log 2>&1
```

### 日常运维命令

| 操作 | 命令 |
|------|------|
| 停止项目 | `crontab -e` → 在两行 cron 前加 `#` 注释 |
| 恢复项目 | `crontab -e` → 去掉 `#` 注释 |
| 手动重扫描 | `python run_scanner.py --force` |
| 手动模拟盘 | `python run_scanner.py --phase 4` |
| 模拟盘 dry-run | `python run_scanner.py --phase 4 --dry-run` |
| 看当前状态 | `cat output/state.json \| python -m json.tool \| head -30` |
| 看今日日志 | `tail -50 logs/daily_$(date +%Y%m%d).log` |
| 看历史日志 | `ls -la logs/` |
| 拉取新代码 | `git pull` |
| 修改配置 | 编辑 `config/scanner_config.py` |

### 重要说明

- **首次运行**：必须先手动 `python run_scanner.py --force` 完成扫描，state.json 进入 `running` 状态后 cron 才能正常执行模拟盘
- **非交易日**：Cron 每天触发，脚本内部调用 baostock API 判断交易日，非交易日自动跳过
- **季度重扫描**：3/6/9/12月1日凌晨执行，距上次扫描 > 90 天也会触发
- **重扫描不影响模拟盘**：新扫描结果会覆盖旧的 Top 10 股票和指标，下一次模拟盘自动使用新结果

---

## 📊 输出结果解读

### state.json 关键字段

| 字段 | 含义 |
|------|------|
| `current_phase` | 当前阶段：`idle` / `scanning` / `selecting` / `verifying` / `running` |
| `best_indicator` | Phase 1 选出的最佳指标 |
| `top_10_stocks` | Phase 2 选出的 Top 10 股票代码 |
| `scan_results` | 全部 97 个指标的评分排行 |
| `portfolio.cash` | 模拟盘当前现金 |
| `portfolio.positions` | 当前持仓（股票代码→股数/成本） |
| `portfolio.pending_orders` | 明日待执行订单（buy/sell/hold） |
| `trade_log` | 历史交易记录（最近 180 天） |

### 日报指标

| 指标 | 含义 |
|------|------|
| 当日操作 | 今日执行的买卖（买入→成本，卖出→盈亏） |
| 持仓摘要 | 每只持仓的股数、成本价、现价、浮动盈亏 |
| 账户摘要 | 总资产、现金、累计收益率 |
| 明日信号 | 基于今日收盘数据生成的明日待执行操作 |

---

## 🔧 配置参数说明

所有参数集中管理在 `config/scanner_config.py`，修改后下次运行生效：

### 扫描参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SCAN_YEARS` | Phase 1 回测期（年） | `2` |
| `VERIFY_MONTHS` | Phase 3 验证期（月） | `3` |
| `RESCAN_DAYS` | 自动重扫间隔（天） | `90` |
| `TOP_N_STOCKS` | 选股数量 | `10` |
| `VERIFY_PASS_THRESHOLD` | 验证通过阈值（N/10） | `6` |
| `STD_PENALTY` | 评分公式标准差惩罚系数 | `0.5` |

### 并行与资金

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MAX_WORKERS` | 并行工作进程数 | `32` |
| `INITIAL_CAPITAL` | 模拟盘初始资金（元） | `1,000,000` |
| `SLIPPAGE` | 滑点 | `0.003`（0.3%） |
| `COMMISSION_RATE` | 佣金比例 | `0.0005`（万分之五） |
| `TAX_RATE` | 印花税（卖出收取） | `0.001`（千分之一） |
| `POSITION_PCT` | 仓位比例 | `0.95`（95%） |

### 风控参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `STOP_LOSS_PCT` | 止损比例 | `0.05`（5%） |
| `STOP_PROFIT_PCT` | 止盈触发比例 | `0.20`（20%） |
| `DRAWDOWN_PCT` | 回落止盈比例 | `0.03`（3%） |

### WxPusher 推送

| 参数 | 说明 |
|------|------|
| `WXPUSHER_TOKEN` | 推送 token |
| `WXPUSHER_UIDS` | 接收用户 UID |
| `WXPUSHER_TOPIC_IDS` | 推送主题 ID |

---

## ❓ 常见问题

### Q1: 为什么扫描结果中 score 是负的？

评分公式的绝对值没有意义，关键是**相对排序**。负分表示在当前行情下，没有任何指标能在多数股票上稳定跑赢持股不动（这是正常的——择时交易在牛市中天然劣势）。我们选的是 97 个指标中"负得最少"的那个。

### Q2: 超额收益是什么？为什么有些 +50% 有些 −90%？

超额收益 = 策略收益 − **该股票自身**的 buy-and-hold 收益。股票跌了 50% 而策略空仓 → 超额 +50%。股票涨了 100% 而策略只赚了 20% → 超额 −80%。这就是为什么下跌股票上超额往往是正的。

### Q3: Phase 3 验证失败了怎么办？

状态自动重置为 `idle`，下次 cron 触发（或手动 `--force`）会自动重新扫描。这确保只用当前市场条件下仍然有效的指标。

### Q4: bash 提示 `source activate zhulei` 不生效？

```bash
# 用完整路径代替
/home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py --force
```

### Q5: Cron 没有触发？

```bash
# 检查 cron 服务是否在运行
systemctl status cron

# 检查 cron 日志
grep -i cron /var/log/syslog | tail -20

# 手动模拟 cron 执行
cd /public/home/hpc/zhulei/superman/quant/code/015_indicator_scanner
/home/zhulei/anaconda3/envs/zhulei/bin/python run_scanner.py --phase 4
```

### Q6: 修改了代码怎么办？

```bash
git pull                          # 拉取最新代码
python run_scanner.py --force     # 如 Phase 1-3 有变，需重扫描
# Phase 4 逻辑变更：cron 下次自动用新代码，无需手动操作
```

### Q7: 数据来源是什么？

| 阶段 | 数据源 |
|------|--------|
| Phase 1-3（扫描/选股/验证） | 本地 CSV（`../data/input/`，交易日晚间更新） |
| Phase 4（每日模拟盘） | **baostock API 直接拉取**（确保数据最新） |
| 交易日判断 | **baostock API** `query_trade_dates()` |
| 沪深300成分股 | **baostock API** `query_hs300_stocks()`（7天缓存） |

### Q8: 为什么 Phase 4 用 API 而 Phase 1-3 用本地文件？

Phase 1-3 需要扫描 300 只股票 × 97 个指标，用 API 太慢。Phase 4 只需 10 只股票，用 API 确保数据最新（本地 CSV 可能因网络故障漏更新）。

### Q9: 会不会有 look-ahead bias（未来函数）？

不会。框架有两层防护：
1. **信号后移一天**：`pos[i]` 反映 `i-1` 日信号，在 `i` 日开盘执行
2. **次日开盘成交**：所有交易以次日开盘价成交，不是当日收盘价

### Q10: 模拟盘和回测的差异大吗？

模拟盘严格执行与回测相同的逻辑（滑点、佣金、印花税、整百股、FIFO），主要差异在于：
- 回测末尾有虚拟卖出（模拟成绩效），模拟盘不强制卖出
- 回测一次跑完，模拟盘每天增量更新持仓状态

---

> **Happy Scanning! 🔍**
