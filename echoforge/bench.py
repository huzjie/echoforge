# -*- coding: utf-8 -*-
"""Benchmarks: skill-driven metric table + monotonicity + migration metrics."""
from __future__ import annotations

from .kit import build_kit
from .core import metrics as M


def run_skill_table(cfg):
    """Aggregate the four monotone metrics vs skill."""
    kit = build_kit(cfg)
    audit = kit["audit"]
    return [audit.audit_skill(s) for s in cfg["bench"]["skills"]]


def summarize(table):
    a, b = table[0], table[-1]
    return {
        "longctx_recall": (a["longctx_recall"], b["longctx_recall"]),
        "routing_utility": (a["routing_utility"], b["routing_utility"]),
        "echo_acc": (a["echo_acc"], b["echo_acc"]),
        "quality": (a["quality"], b["quality"]),
    }


def check_monotone(table):
    """Verify the four headline metrics are non-decreasing across skill."""
    def inc(get):
        return all(get(table[i]) <= get(table[i + 1]) + 1e-9
                   for i in range(len(table) - 1))

    return {
        "recall_inc": inc(lambda r: r["longctx_recall"]),
        "routing_inc": inc(lambda r: r["routing_utility"]),
        "echo_inc": inc(lambda r: r["echo_acc"]),
        "quality_inc": inc(lambda r: r["quality"]),
    }


def run_migration_bench(cfg):
    """Architectural constants measured from the reference geometry."""
    kit = build_kit(cfg)
    world = kit["world"]
    attn = cfg["attention"]
    echo = cfg["echo"]
    full = M.aggregate(world, 1.0)
    return {
        "memory_ratio": full["memory_ratio"],
        "memory_reduction_pct": round((1.0 - full["memory_ratio"]) * 100, 2),
        "k_steps": echo["k_steps"],
        "decode_speedup_at_skill_1": full["decode_speedup"],
        "param_overhead": full["param_overhead"],
        "seq_len": attn["seq_len"],
        "n_experts": cfg["moe"]["n_experts"],
    }
