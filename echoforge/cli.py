# -*- coding: utf-8 -*-
"""Command-line interface: doctor / bench / modes / retrieve / route / echo / train / serve."""
from __future__ import annotations

import argparse
import json

from .config import load_config
from .kit import build_kit
from . import bench as benchmod
from .core import metrics as M
from .core import attention, moe, echo as echo_mod
from .training.trainer import RewardTrainer
from .tasks import list_tasks


def cmd_doctor(cfg):
    kit = build_kit(cfg)
    world = kit["world"]
    r = kit["audit"].audit_skill(1.0)
    print("[doctor] sequences=%d seq_len=%d experts=%d top_k=%d k_steps=%d"
          % (world.n_sequences, world.seq_len, cfg["moe"]["n_experts"],
             cfg["moe"]["top_k"], cfg["echo"]["k_steps"]))
    print("[doctor] longctx_recall=%.4f routing_utility=%.4f echo_acc=%.4f quality=%.4f"
          % (r["longctx_recall"], r["routing_utility"], r["echo_acc"], r["quality"]))
    ok = (r["longctx_recall"] >= 0.9999 and r["routing_utility"] >= 0.9999
          and r["echo_acc"] >= 0.9999 and r["quality"] >= 0.9999)
    if not ok:
        raise SystemExit("[doctor] FAIL: metrics not at 1.0 at skill=1.0")
    print("[doctor] OK (fully aligned model reproduced all four monotone metrics)")


def cmd_bench(cfg):
    table = benchmod.run_skill_table(cfg)
    print("[bench] skill  recall  routing  echo   quality  speedup")
    for r in table:
        print("[bench] %.2f   %.4f  %.4f   %.4f  %.4f  %.2f"
              % (r["skill"], r["longctx_recall"], r["routing_utility"],
                 r["echo_acc"], r["quality"], r["decode_speedup"]))
    s = benchmod.summarize(table)
    print("[bench] recall %.4f->%.4f | routing %.4f->%.4f | echo %.4f->%.4f | quality %.4f->%.4f"
          % (s["longctx_recall"][0], s["longctx_recall"][1],
             s["routing_utility"][0], s["routing_utility"][1],
             s["echo_acc"][0], s["echo_acc"][1],
             s["quality"][0], s["quality"][1]))
    mono = benchmod.check_monotone(table)
    print("[bench] monotone: %s" % ", ".join("%s=%s" % (k, v) for k, v in mono.items()))
    mig = benchmod.run_migration_bench(cfg)
    print("[bench] memory_ratio=%.4f (-%.2f%% KV) decode_speedup=%.2fx param_overhead=%.6f"
          % (mig["memory_ratio"], mig["memory_reduction_pct"],
             mig["decode_speedup_at_skill_1"], mig["param_overhead"]))


def cmd_modes(cfg):
    kit = build_kit(cfg)
    world = kit["world"]
    print("[modes] sequences=%d seq_len=%d skill=%.2f"
          % (world.n_sequences, world.seq_len, cfg["model"]["skill"]))
    print("[modes] capabilities=%s" % ", ".join(t["id"] for t in list_tasks()))


def cmd_retrieve(cfg, args):
    kit = build_kit(cfg)
    world = kit["world"]
    n_global, n_routers, rank = M.skill_params(cfg, args.skill)
    retained = attention.retained_positions(world.seq_len, cfg["attention"]["window"], n_global)
    recall = attention.needle_recall(world.needles, retained)
    print("[retrieve] skill=%.2f n_global=%d retained_positions=%s"
          % (args.skill, n_global, sorted(retained)))
    print("[retrieve] needle_recall=%.4f (needles=%s)" % (recall, world.needles))


def cmd_route(cfg, args):
    token = args.token % 256
    moe_cfg = cfg["moe"]
    n_routers = max(1, round(1 + args.skill * (moe_cfg["max_routers"] - 1)))
    scores = moe.expert_scores(token, moe_cfg["n_experts"], n_routers, moe_cfg.get("eps", 1.0))
    top = moe.select_topk(scores, moe_cfg["top_k"])
    g = token % moe_cfg["n_experts"]
    print("[route] token=%d true_expert=%d routers=%d hit=%s"
          % (token, g, n_routers, g in top))
    print("[route] top-%d=%s" % (moe_cfg["top_k"], sorted(top)))


def cmd_echo(cfg, args):
    token = args.token % 256
    rank = round(args.skill * cfg["echo"]["max_rank"])
    h = echo_mod.hidden_state(token)
    preds = echo_mod.echo_predict(h, rank)
    truth = echo_mod.next_k(token, cfg["echo"]["k_steps"])
    print("[echo] token=%d rank=%d" % (token, rank))
    for j in range(cfg["echo"]["k_steps"]):
        print("[echo] step%d pred=%s truth=%d" % (j + 1, preds[j], truth[j]))


def cmd_train(cfg, args):
    kit = build_kit(cfg)
    trainer = RewardTrainer(kit["audit"], skill=args.start, lr=args.lr)
    for _ in range(args.steps):
        rec = trainer.step()
        print("[train] step %02d skill=%.4f reward=%.4f delta=%.4f"
              % (rec["step"], rec["skill"], rec["reward"], rec["delta"]))
    print("[train] reward monotone=%s" % trainer.is_monotone())


def cmd_serve(cfg, args):
    from .serving.app import serve
    serve(host=args.host, port=args.port, cfg=cfg)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="echoforge",
        description="Long-context HCA attention + multi-router MoE + ECHO multi-step prediction "
                    "(ERNIE 5.5 Flash Base style) reference implementation")
    ap.add_argument("--recipe", default=None, help="yaml recipe to merge over defaults")
    sub = ap.add_subparsers(dest="cmd")

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--recipe", default=None, help="yaml recipe to merge over defaults")

    sub.add_parser("doctor", parents=[parent])
    sub.add_parser("bench", parents=[parent])
    sub.add_parser("modes", parents=[parent])

    p = sub.add_parser("retrieve", parents=[parent])
    p.add_argument("--skill", type=float, default=0.5)

    p = sub.add_parser("route", parents=[parent])
    p.add_argument("--token", type=int, default=0)
    p.add_argument("--skill", type=float, default=0.5)

    p = sub.add_parser("echo", parents=[parent])
    p.add_argument("--token", type=int, default=0)
    p.add_argument("--skill", type=float, default=0.5)

    p = sub.add_parser("train", parents=[parent])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--start", type=float, default=0.0)

    p = sub.add_parser("serve", parents=[parent])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8431)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    cfg = load_config(args.recipe)

    if args.cmd is None:
        ap.print_help()
        return
    if getattr(args, "skill", None) is None:
        args.skill = cfg["model"]["skill"]

    dispatch = {
        "doctor": lambda: cmd_doctor(cfg),
        "bench": lambda: cmd_bench(cfg),
        "modes": lambda: cmd_modes(cfg),
        "retrieve": lambda: cmd_retrieve(cfg, args),
        "route": lambda: cmd_route(cfg, args),
        "echo": lambda: cmd_echo(cfg, args),
        "train": lambda: cmd_train(cfg, args),
        "serve": lambda: cmd_serve(cfg, args),
    }
    dispatch[args.cmd]()


if __name__ == "__main__":
    main()
