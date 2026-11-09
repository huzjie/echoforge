# 快速开始

echoforge 是一个零依赖的纯 Python 参考实现，Python 3.8+ 即可运行。

## 安装

无需安装任何第三方依赖，克隆即用：

```bash
git clone https://github.com/huzjie/echoforge.git
cd echoforge
```

## 自检

```bash
python -m echoforge doctor
```

`doctor` 会在 `skill=1.0`（完全对齐）下跑一遍四个指标，四个必须全部命中 1.0：

```
[doctor] longctx_recall=1.0000 routing_utility=1.0000 echo_acc=1.0000 quality=1.0000
[doctor] OK (fully aligned model reproduced all four monotone metrics)
```

## 跑基准

```bash
python -m echoforge bench
```

输出四项单调曲线 + 内存/加速迁移指标。

## 逐项查看

```bash
# 长上下文缓存保留与 needle 召回
python -m echoforge retrieve --skill 0.5

# MoE 专家选择（top-k + 是否命中真值专家）
python -m echoforge route --token 10 --skill 0.5

# ECHO 多步预测
python -m echoforge echo --token 10 --skill 0.5

# 训练（单调 skill 轴）
python -m echoforge train --steps 10
```

## 用配方调参

```bash
python -m echoforge bench --recipe recipes/full.yaml
```

`recipes/` 下是五套预设：`minimal.yaml`（最小规模）、`dense.yaml`（大词表）、
`low_rank.yaml`（低秩 MLA）、`fast.yaml`（大 ECHO 步长）、`full.yaml`（完整）。

## 启动 HTTP API

```bash
python -m echoforge serve --port 8431
```

```bash
curl http://127.0.0.1:8431/health
curl -X POST http://127.0.0.1:8431/v1/run -d '{"skill": 1.0}'
```

## 运行测试

```bash
python -m unittest discover -s tests -t .
```
