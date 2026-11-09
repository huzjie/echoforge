# -*- coding: utf-8 -*-
"""Zero-dependency HTTP API over the echoforge model (http.server).

Endpoints:
  GET  /health      -> liveness
  GET  /v1/info     -> corpus + model metadata
  POST /v1/run      -> full four-metric audit for {skill}
  POST /v1/retrieve -> HCA cache retention + needle recall for {skill}
  POST /v1/route    -> expert selection for {token, skill}
  POST /v1/echo     -> multi-step predictions for {token, skill}
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..kit import build_kit
from ..core import metrics as M
from ..core import attention, moe, echo as echo_mod


def build_app(cfg):
    kit = build_kit(cfg)
    world = kit["world"]
    skill_default = cfg["model"]["skill"]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "service": "echoforge"})
            if self.path == "/v1/info":
                return self._send(200, {
                    "seq_len": world.seq_len,
                    "n_sequences": world.n_sequences,
                    "n_experts": cfg["moe"]["n_experts"],
                    "top_k": cfg["moe"]["top_k"],
                    "k_steps": cfg["echo"]["k_steps"],
                    "default_skill": skill_default,
                })
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                return self._send(400, {"error": "bad json"})
            skill = float(data.get("skill", skill_default))

            if self.path == "/v1/run":
                return self._send(200, kit["audit"].audit_skill(skill))
            if self.path == "/v1/retrieve":
                n_global, _, _ = M.skill_params(cfg, skill)
                retained = attention.retained_positions(
                    world.seq_len, cfg["attention"]["window"], n_global)
                return self._send(200, {
                    "skill": skill, "n_global": n_global,
                    "retained_positions": sorted(retained),
                    "needle_recall": attention.needle_recall(world.needles, retained),
                })
            if self.path == "/v1/route":
                token = int(data.get("token", 0)) % 256
                mc = cfg["moe"]
                n_routers = max(1, round(1 + skill * (mc["max_routers"] - 1)))
                scores = moe.expert_scores(token, mc["n_experts"], n_routers, mc.get("eps", 1.0))
                top = moe.select_topk(scores, mc["top_k"])
                return self._send(200, {
                    "token": token, "true_expert": token % mc["n_experts"],
                    "n_routers": n_routers, "top_k": sorted(top),
                    "hit": (token % mc["n_experts"]) in top,
                })
            if self.path == "/v1/echo":
                token = int(data.get("token", 0)) % 256
                rank = round(skill * cfg["echo"]["max_rank"])
                preds = echo_mod.echo_predict(echo_mod.hidden_state(token), rank)
                return self._send(200, {
                    "token": token, "rank": rank, "pred": preds,
                    "truth": echo_mod.next_k(token, cfg["echo"]["k_steps"]),
                })
            return self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    return Handler


def serve(host="127.0.0.1", port=8431, cfg=None):
    if cfg is None:
        from ..config import load_config
        cfg = load_config()
    handler = build_app(cfg)
    server = HTTPServer((host, port), handler)
    print("[serve] echoforge API listening on http://%s:%d" % (host, port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
