# 基准说明

## 度量定义

四项单调指标（0 → 1，随 `skill` 单调非递减）：

| 指标 | 定义 | 判真方式 |
| --- | --- | --- |
| `longctx_recall` | 被 HCA 缓存保留的 needle 比例 | needle 位置是否在缓存集合内 |
| `routing_utility` | 真值专家进入 top-k 的 token 比例 | `t % E` 是否在 top-k 中 |
| `echo_acc` | ECHO 低秩头正确预测的未来步比例 | 预测 token == `next^j(t)` |
| `quality` | 三轴同时诚实的轮次比例 | recall ∧ routing ∧ echo |

迁移指标（架构常量 / 测量值）：

| 指标 | 含义 |
| --- | --- |
| `memory_ratio` | HCA+MLA 平均每层 KV 相对 full attention 的比值（越低越好） |
| `decode_speedup` | `echo_acc × k_steps`，每前向有效产出 token 数 |
| `param_overhead` | ECHO 额外参数占 K 个独立头的比例 |

## 复现

```bash
python -m echoforge bench
```

默认 `n_sequences=8`、`seq_len=64`、`n_experts=64`、`top_k=8`、`k_steps=4`，
实测结果（`config.py` 默认值）：

| skill | recall | routing | echo | quality | speedup |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.0000 | 0.6953 | 0.0000 | 0.0000 | 0.00x |
| 0.25 | 0.2500 | 0.9766 | 0.2500 | 0.0547 | 1.00x |
| 0.50 | 0.5000 | 1.0000 | 0.5000 | 0.2500 | 2.00x |
| 0.75 | 0.7500 | 1.0000 | 0.7500 | 0.5625 | 3.00x |
| 1.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.00x |

- 单调性：`recall_inc / routing_inc / echo_inc / quality_inc` 全部 `True`
- `memory_ratio = 0.1133`（-88.67% KV 显存）
- `param_overhead = 0.0039`

> 说明：`routing_utility` 的基线不是 0，因为即使单路由器，`+1.0` 的真实信号也能
> 让真值专家在 top-8/64 中保持约 70% 的命中（纯随机只有 8/64 = 12.5%）。这正是
> 单路由器"能用但不稳"、多路由器"稳"的直观体现。
