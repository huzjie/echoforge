# echoforge — 长上下文混合注意力 + 多路由器 MoE + ECHO 多步预测

> 百度 **ERNIE 5.5 Flash Base**（280B 稀疏 MoE 基座）预训练架构三件套的零依赖参考实现。
> 纯 Python、零第三方依赖、全流程可复现，用「填完配置即可跑的代码」讲清三个可复用工程创新背后的技术原理。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#)
[![Zero deps](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#)
[![Tests](https://img.shields.io/badge/tests-99%20passed-green.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## 一句话讲清楚

一个 280B 的稀疏 MoE 大模型，凭什么既吃得住长上下文、又跑得快、路由还准？ERNIE 5.5 Flash Base 公开的三个架构手段，各自解决一个具体瓶颈，而且**每一个都能拆出来单独复用**：

| 瓶颈 | 手段 | 本仓库对应实现 |
| --- | --- | --- |
| 长上下文 KV 显存爆炸 | **HCA 混合注意力 + MLA 潜变量压缩** | `core/attention.py` |
| MoE 路由评分与真实效用不相关 | **Multi-Router Latent MoE（多路由器投票）** | `core/moe.py` |
| 逐 token 解码太慢 | **ECHO 多步预测（复用末层一次出多个 token）** | `core/echo.py` |

本仓库不是"复述论文"，而是把每个手段做成**确定性的、可验证的、闭式判真的最小闭环**——每个模块都能单独跑、单独测、单独看指标曲线。

---

## 技术原理拆解

### 1. HCA 混合注意力：长上下文为什么要"按层分级缓存"

标准 Transformer 每层、每个 token 都要存 K/V，显存随上下文长度线性增长，这就是长上下文吃显存的根因。

ERNIE 5.5 的做法是**分层分级**（Hybrid Cache Attention）：

- **每 8 层只有 1 层保留全上下文**（用 MLA，即 Multi-head Latent Attention，把 K/V 压到低秩潜变量，显存从 `2d` 降到 `2r`）；
- **其余 7 层跑 HCA**：只保留「最近 `window` 个 token 的滑动窗口」+「一个固定大小的全局摘要缓存」，其余历史直接丢弃。

代价是诚实的——**缓存里没保留的历史 token，之后就没法再被检索到**。本仓库把这个代价量化成 `longctx_recall`（needle 检索命中率），并测出显存压缩比：

```
实测 memory_ratio = 0.1133  →  KV 显存 -88.67%
（64 上下文、dim=32、MLA rank=8、window=8、global=16、full_ctx_every=8）
```

### 2. Multi-Router Latent MoE：路由为什么"多路由器投票"更准

单路由器给每个专家打分，分数 = 真实信号 + 噪声。噪声大时，真正该选的专家会被噪声挤出 top-k。

**多路由器**就是集成里的方差缩减：`R` 个独立路由器打分求平均，噪声幅度按 `1/√R` 衰减。本仓库把这个关系显式建模成 `noise_scale(R) = eps / √R`，于是"命中真值专家"的概率**严格随路由器数量单调上升**：

```
实测 routing_utility：单路由器 0.6953 → 8 路由器 1.0000（单调）
```

### 3. ECHO 多步预测：为什么"多预测几个 token"几乎不增参数

普通做法是给「预测第 2、3、4 个 token」各挂一个 `V×d` 的独立头，参数随步数线性膨胀。

ECHO 的关键是**复用主干末层的隐状态**：隐状态里其实已经编码了未来信息，只需要一个**低秩投影**（`rank×d` 参数）就能同时读出未来 `k` 个 token。投影秩就是诚实的容量开关——秩不够的步读不出来（返回 `None`）：

```
实测 echo_acc = min(rank, K)/K：rank 0→4 对应 0 → 1.0（单调）
实测 decode_speedup = 4.0x（一次前向出 4 个 token）
实测 param_overhead = 0.0039（ECHO 额外参数仅占 K 个独立头的 0.39%）
```

---

## 可复用思路（干货块）

这套实现里藏着几个可以直接搬到生产项目的工程技巧：

1. **任何"打分 + 选 top-k"都能用集成降噪**：路由、召回、检索重排……只要打分带独立噪声，多模型/多路由器投票平均就是免费的方差缩减（`1/√R`）。本仓库 `moe.py` 里的 `noise_scale` 是可直接复用的公式。

2. **长上下文缓存是"容量换召回"的显式权衡**：先确定预算（`window` + `global`），再接受"预算外的历史不可检索"。做 RAG/长文档时同理——先定 chunk 预算，再谈命中率。

3. **多步预测用"复用隐状态 + 低秩头"替代"每步独立头"**：参数从 `K·V·d` 降到 `rank·d`，`echo.py` 的 `echo_param_overhead` 就是成本公式。

4. **闭式 oracle + 程序化校验 = 可验证奖励**：本仓库的 next-token（`(7t+13) mod 256`）、真值专家（`t mod E`）、未来 token 全是纯函数，正确性可精确判定，RL 收敛方向与真值严格对齐——零标注、零奖励模型。

5. **确定性 RNG 用 `md5(key)` 前 8 字节做种子**：全项目逐位可复现，跨平台、跨进程结果一致（见 `utils/rng.py`）。

6. **四项单调指标 = 三条独立诚实轴组合**：缓存容量 / 路由器数 / 投影秩各自只作用一个方面，指标取「单轴/单轴/单轴/三轴∧」，得到 recall / routing / echo / quality 四条互不重复且严格单调的 0→1 曲线（见 `core/metrics.py`）。

---

## 快速开始

```bash
# 零依赖，Python 3.8+ 即可
git clone https://github.com/huzjie/echoforge.git
cd echoforge

# 自检：四项指标在 skill=1.0 时必须全部命中
python -m echoforge doctor

# 跑完整基准：四项单调曲线 + 内存/加速迁移指标
python -m echoforge bench

# 看某个 token 的多步预测
python -m echoforge echo --token 10 --skill 0.5

# 看某条序列的缓存保留与 needle 召回
python -m echoforge retrieve --skill 0.5
```

也可以直接填配置：`recipes/*.yaml` 是五套预设（`--recipe recipes/full.yaml`），`config.py` 的 `DEFAULTS` 是全部可调项。

---

## 基准结果（真实测量，非估算）

`n_sequences=8`、`seq_len=64`、`n_experts=64`、`top_k=8`、`k_steps=4`：

| skill | longctx_recall | routing_utility | echo_acc | quality | decode_speedup |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.0000 | 0.6953 | 0.0000 | 0.0000 | 0.00x |
| 0.25 | 0.2500 | 0.9766 | 0.2500 | 0.0547 | 1.00x |
| 0.50 | 0.5000 | 1.0000 | 0.5000 | 0.2500 | 2.00x |
| 0.75 | 0.7500 | 1.0000 | 0.7500 | 0.5625 | 3.00x |
| 1.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.00x |

- 四项指标全部**单调非递减**（`recall_inc / routing_inc / echo_inc / quality_inc = True`）
- `memory_ratio = 0.1133`（KV 显存 **-88.67%**）
- `param_overhead = 0.0039`（ECHO 额外参数 **0.39%**）
- 99 个单元测试全绿，`compileall` 零错误

---

## 架构与目录

```
echoforge/
  core/          # 科学内核：attention / moe / echo / oracle / metrics / validate
  world/         # 确定性长上下文语料 + 三轴指标跑器
  training/      # 可验证奖励审计 + 单调 skill 轴
  tasks/         # 能力目录（retrieve / route / echo / bench / speedup）
  serving/       # 零依赖 HTTP API
  utils/         # 确定性 RNG / 哈希 / 线性代数 / 极简 YAML 子集
tests/           # 99 个单元测试
docs/            # 7 篇技术文档
prompts/         # 6 份提示词模板
recipes/         # 5 份配置配方
examples/        # 4 个可运行示例
deploy/          # Docker / K8s / Helm
```

详细拆解见 [`docs/architecture.md`](docs/architecture.md)，提示词模板见 [`prompts/`](prompts/)，配方见 [`recipes/`](recipes/)。

---

## License

MIT © ai-daily-agent
