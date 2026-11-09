# API 参考

## Python API

```python
from echoforge.config import load_config
from echoforge.kit import build_kit

cfg = load_config()                 # 或 load_config("recipes/full.yaml")
kit = build_kit(cfg)
world = kit["world"]
audit = kit["audit"]

m = audit.audit_skill(0.5)          # 四项指标 + 迁移指标
# {'longctx_recall': ..., 'routing_utility': ..., 'echo_acc': ...,
#  'quality': ..., 'memory_ratio': ..., 'decode_speedup': ..., 'param_overhead': ...}

world.params_at(0.5)                # (n_global, n_routers, rank)
```

### core 模块

| 函数 | 位置 | 说明 |
| --- | --- | --- |
| `token_embed(t)` | `core.embed` | 确定性单位嵌入 |
| `next_token(t)` / `next_k(t,k)` | `core.oracle` | 闭式下一 token / 未来 k 个 token |
| `true_expert(t,E)` | `core.oracle` | 闭式真值专家 |
| `retained_positions(L,w,g)` | `core.attention` | HCA 缓存保留的位置集合 |
| `needle_recall(n, retained)` | `core.attention` | needle 召回 |
| `memory_ratio(...)` | `core.attention` | KV 显存压缩比 |
| `expert_scores(t,E,R)` | `core.moe` | 路由器打分 |
| `select_topk(scores,k)` | `core.moe` | top-k 选择 |
| `routing_utility(tokens,...)` | `core.moe` | 真值专家命中率 |
| `hidden_state(t)` | `core.echo` | 末层隐状态 |
| `echo_predict(h, rank)` | `core.echo` | 低秩多步预测 |
| `echo_accuracy(tokens, rank)` | `core.echo` | 多步命中率 |

## HTTP API

启动：`python -m echoforge serve --port 8431`

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活检查 |
| GET | `/v1/info` | 语料与模型元信息 |
| POST | `/v1/run` | `{"skill": 0.5}` → 四项指标 |
| POST | `/v1/retrieve` | `{"skill": 0.5}` → 缓存保留 + needle 召回 |
| POST | `/v1/route` | `{"token": 10, "skill": 0.5}` → 专家选择 |
| POST | `/v1/echo` | `{"token": 10, "skill": 0.5}` → 多步预测 |

## CLI

`doctor` / `bench` / `modes` / `retrieve` / `route` / `echo` / `train` / `serve`，
详见 `python -m echoforge --help`。
