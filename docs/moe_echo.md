# 多路由器 MoE + ECHO 多步预测

## Multi-Router Latent MoE

### 问题

MoE 路由器的打分 = 真实信号 + 噪声。噪声一大，真正该选的专家被挤出 top-k，出现
"路由评分高、实际效用低"的错配。

### 方案：多路由器投票

`R` 个独立路由器打分求平均，噪声按 `1/√R` 衰减。`core/moe.py` 显式建模：

```python
noise_scale(R) = eps / sqrt(R)
score(e)       = (1.0 if e == true_expert else 0.0) + noise_scale(R) * z[e]
```

`z[e]` 是 `md5(token, e)` 派生的固定噪声。于是真值专家被选进 top-k 的概率**严格随
路由器数量单调上升**：单路由器 `routing_utility = 0.6953`，8 路由器 `1.0000`。

## ECHO 多步预测

### 问题

预测第 2、3、4 个 token 若各挂一个 `V×d` 独立头，参数随步数线性膨胀。

### 方案：复用末层隐状态 + 低秩投影

末层隐状态已经编码了未来信息。ECHO 用一个 `rank×d` 的低秩投影同时读出未来 `k` 个
token，`rank` 就是诚实容量开关：

```python
echo_predict(h, rank) -> [pred_1, ..., pred_k]   # 步 j ≥ rank 时返回 None
echo_param_overhead(rank) = rank*d / (k*vocab*d)  # 默认 0.0039
```

`echo_acc = min(rank, K)/K` 单调 0 → 1，`decode_speedup = echo_acc × K`，满秩时一次
前向出 4 个 token（4.0x），额外参数仅 0.39%。

## 可复用要点

- 任何"打分 + top-k"都能用集成降噪（`1/√R`）；
- 多步预测复用隐状态比"每步独立头"省 `K·V·d → rank·d` 的参数；
- 低秩瓶颈是显式的"容量-准确率"旋钮，可做蒸馏与量化。
