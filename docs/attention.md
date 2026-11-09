# HCA 混合注意力 + MLA 潜变量压缩（长上下文显存）

## 问题

标准 Transformer 每一层、每一个 token 都要存 key 和 value，每层 KV 显存为
`2 · seq_len · dim`。上下文一长，显存直接线性爆掉。这是长上下文的第一瓶颈。

## ERNIE 5.5 的分层分级方案

**MLA（Multi-head Latent Attention）**：把 K/V 压到低秩潜变量 `r`，一个全上下文层
的 KV 从 `2·seq_len·dim` 降到 `2·seq_len·r`。

**HCA（Hybrid Cache Attention）**：不必每层都全上下文。每 `full_ctx_every` 层只有
1 层保留全上下文（MLA），其余层跑 HCA——只保留「滑动窗口 `window`」+「全局摘要
`global`」，其余历史丢弃。

## 本仓库的实现

`core/attention.py` 用纯函数刻画这个内存账本：

```python
retained_positions(seq_len, window, n_global)
#   = {最近 window 个位置} ∪ {最早 n_global 个位置}

memory_ratio(seq_len, dim, latent_rank, window, n_global, full_ctx_every)
#   = (1/f · mla_kv + (f-1)/f · hca_kv) / full_attention_kv
```

默认参数下 `memory_ratio = 0.1133`，即 KV 显存 **-88.67%**。

## 诚实的代价

HCA 丢弃历史是有代价的：**缓存里没有的 token，之后再也检索不到**。这个代价被量化
成 `needle_recall`——把"关键 token"埋在最前面（needle），看缓存容量多大时才能把它
捞回来：

```bash
python -m echoforge retrieve --skill 0.5
```

`skill` 越大，`n_global` 越大，保留的 needle 越多，`longctx_recall` 单调升到 1.0。

## 可复用要点

- 长上下文 = 「容量换召回」的显式权衡，先定预算再谈命中；
- 分层缓存是通用的显存优化思路：不必每层都付全上下文的钱；
- 低秩压缩（MLA）与缓存裁剪（HCA）正交，可叠加。
