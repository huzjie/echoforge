# 架构总览

echoforge 把 ERNIE 5.5 Flash Base 的三个架构创新各自做成一个**可独立验证的闭环**，
再用一个 `skill` 轴把它们串起来，形成四条严格单调的指标。

## 三层结构

```
┌─────────────────────────────────────────────────────────────┐
│                     EchoWorld (world/)                       │
│   确定性长上下文语料 + 三轴指标跑器（把 skill 映射到三个旋钮）│
├─────────────────────────────────────────────────────────────┤
│  core/attention.py   core/moe.py   core/echo.py              │
│  HCA+MLA 混合注意力    多路由器 MoE      ECHO 多步预测          │
│      ↓                    ↓                 ↓                │
│  core/validate.py —— 闭式判真（needle/专家/未来 token）      │
│  core/metrics.py  —— 四项单调指标 + 迁移指标                 │
├─────────────────────────────────────────────────────────────┤
│  training/ —— 可验证奖励审计 + 单调 skill 轴                │
│  serving/  —— 零依赖 HTTP API                                │
└─────────────────────────────────────────────────────────────┘
```

## skill 轴：三条独立诚实轴

`skill ∈ [0,1]` 是"模型与闭式真值的对齐度"，它驱动三个互不干扰的旋钮：

| 旋钮 | 计算 | 诚实轴 | 指标 |
| --- | --- | --- | --- |
| HCA 全局缓存容量 | `n_global = round(skill * max_global)` | 注意力 | `longctx_recall` |
| 路由器数量 | `n_routers = 1 + round(skill * (max_routers-1))` | MoE | `routing_utility` |
| ECHO 投影秩 | `rank = round(skill * max_rank)` | 多步预测 | `echo_acc` |

## 四项指标为何互不重复

每个旋钮只影响一个方面，指标取「单轴 / 单轴 / 单轴 / 三轴∧」：

- `longctx_recall` —— 单轴（缓存容量）
- `routing_utility` —— 单轴（路由器数）
- `echo_acc` —— 单轴（投影秩）
- `quality` —— 三轴同时诚实（AND 瓶颈）

于是四条曲线严格单调且互不重复，`skill=1.0` 时全部命中 1.0。

## 确定性保证

所有随机性都来自 `md5(key)` 前 8 字节种子（`utils/rng.py`），闭式 oracle
（`core/oracle.py`）是纯函数。同一个配置在任何平台、任何进程上逐位复现。

## 与真实模型的关系

本仓库是**协议形态一致、规模缩小的参考实现**：真实 ERNIE 5.5 Flash Base 是 280B
参数、每层 512 专家取 top-10+1 的稀疏 MoE；本仓库用 64 专家、top-8、64 长上下文
在笔记本上亚秒级跑完，但 HCA/MLA/Multi-Router/ECHO 的**数学结构完全保留**，方便
逐模块读懂、逐模块复用。
