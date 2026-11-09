# -*- coding: utf-8 -*-
"""Metric aggregation: four monotone metrics + three migration metrics.

The four headline metrics are *distinct* because they are built from three
independent honesty axes, each a deterministic, monotone function of ``skill``:

* **longctx_recall** — HCA global-cache capacity ``n_global`` (attention axis);
* **routing_utility** — multi-router ensemble size ``n_routers`` (MoE axis);
* **echo_acc** — ECHO projection rank ``rank`` (multi-step prediction axis);
* **quality** — all three axes honest simultaneously (the AND bottleneck).

The migration metrics are architectural constants/measurements:

* **memory_ratio** — HCA+MLA KV budget relative to full attention (lower better);
* **decode_speedup** — effective tokens per forward pass (``echo_acc * K``);
* **param_overhead** — ECHO extra params vs K independent heads (tiny).
"""
from __future__ import annotations

from ..constants import N_ROUNDS
from .attention import retained_positions, memory_ratio as _memory_ratio
from .moe import router_hits_true
from .echo import echo_step_correct, echo_param_overhead


def skill_params(cfg: dict, skill: float):
    """Map the alignment skill in [0,1] to the three honesty-axis knobs."""
    skill = max(0.0, min(1.0, float(skill)))
    attn = cfg["attention"]
    moe = cfg["moe"]
    echo = cfg["echo"]
    n_global = round(skill * attn["max_global"])
    n_routers = max(1, round(1 + skill * (moe["max_routers"] - 1)))
    rank = round(skill * echo["max_rank"])
    return n_global, n_routers, rank


def aggregate(world, skill: float) -> dict:
    """Aggregate the four monotone metrics across the whole corpus at one skill."""
    cfg = world.cfg
    n_global, n_routers, rank = skill_params(cfg, skill)
    attn = cfg["attention"]
    moe = cfg["moe"]
    echo = cfg["echo"]
    eps = moe.get("eps", 1.0)

    retained = retained_positions(attn["seq_len"], attn["window"], n_global)
    needles = world.needles  # positions 0..N_ROUNDS-1, per sequence
    k_steps = echo["k_steps"]

    rec = rout = echoacc = qual = 0.0
    n = 0
    for seq in world.sequences:
        for r in range(N_ROUNDS):
            r_i = needles[r] in retained
            u_i = router_hits_true(seq[r], moe["n_experts"], moe["top_k"], n_routers, eps)
            e_i = echo_step_correct(seq[r], r % k_steps, rank, k_steps)
            rec += r_i
            rout += u_i
            echoacc += e_i
            qual += 1.0 if (r_i and u_i and e_i) else 0.0
            n += 1

    n = max(1, n)
    return {
        "longctx_recall": rec / n,
        "routing_utility": rout / n,
        "echo_acc": echoacc / n,
        "quality": qual / n,
        "memory_ratio": _memory_ratio(
            attn["seq_len"], attn["dim"], attn["mla_rank"], attn["window"],
            n_global, attn["full_ctx_every"]),
        "decode_speedup": (echoacc / n) * k_steps,
        "param_overhead": echo_param_overhead(rank, attn["dim"], k_steps=k_steps),
    }
