# -*- coding: utf-8 -*-
"""Declarative task catalogue: the three echoforge capabilities."""
from __future__ import annotations

TASKS = [
    {
        "id": "retrieve",
        "name": "长上下文检索",
        "desc": "HCA 混合注意力用「滑动窗口 + 全局摘要」压缩 KV，代价是可检索的历史由缓存容量决定。",
    },
    {
        "id": "route",
        "name": "多路由器 MoE",
        "desc": "多路由器投票选择 top-k 专家，平均掉单路由器噪声，提升路由评分与真实效用的相关性。",
    },
    {
        "id": "echo",
        "name": "ECHO 多步预测",
        "desc": "复用主干末层隐状态，用低秩投影一次预测未来 k 个 token，替代逐 token 解码。",
    },
    {
        "id": "bench",
        "name": "可验证基准",
        "desc": "以 skill 驱动三轴（缓存/路由器/秩），聚合 recall/routing/echo/quality 四项单调指标。",
    },
    {
        "id": "speedup",
        "name": "内存与加速",
        "desc": "度量 HCA+MLA 的 KV 显存压缩比与 ECHO 的每前向有效产出 token 数。",
    },
]


def list_tasks():
    return TASKS


def get_task(task_id):
    for t in TASKS:
        if t["id"] == task_id:
            return t
    return None
