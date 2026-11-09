# -*- coding: utf-8 -*-
"""Verifiable reward: the skill-driven audit of the echoforge model.

There is no learned reward model and no human labelling. For every ``skill`` the
model is driven with *three* independent honesty axes, each a deterministic,
monotone function of ``skill``:

* **attention** — HCA global-cache capacity (``n_global``);
* **MoE**      — multi-router ensemble size (``n_routers``);
* **ECHO**     — multi-step projection rank (``rank``).

The produced behaviour is then scored by the real closed-form checks, giving four
metrics that are monotone 0 -> 1 in ``skill`` *by construction*:

* **longctx_recall**  — needles kept in cache (attention);
* **routing_utility** — true expert survives top-k (MoE);
* **echo_acc**        — future step predicted (ECHO);
* **quality**         — all three axes honest simultaneously (AND bottleneck).
"""
from __future__ import annotations


class EchoAudit:
    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world

    def audit_skill(self, skill):
        """Aggregate the four monotone metrics + migration metrics at one skill."""
        m = self.world.run(skill)
        m["skill"] = skill
        return m

    def mean_reward(self, skill):
        """The scalar training signal: overall quality (monotone in skill)."""
        return self.world.run(skill)["quality"]

    def speedup(self, skill=1.0):
        """Effective tokens-per-forward of the ECHO head at a given skill."""
        return self.world.run(skill)["decode_speedup"]

    def memory_ratio(self, skill=1.0):
        return self.world.run(skill)["memory_ratio"]
