# -*- coding: utf-8 -*-
"""Zero-dependency HTTP client demo (urllib only).

Start the server first:  python -m echoforge serve --port 8431
"""
from __future__ import annotations

import json
import urllib.request


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:8431" + path, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8431" + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    print("health:", get("/health"))
    print("info:", get("/v1/info"))
    print("run:", post("/v1/run", {"skill": 1.0}))
    print("echo:", post("/v1/echo", {"token": 10, "skill": 0.5}))
    print("route:", post("/v1/route", {"token": 10, "skill": 0.5}))


if __name__ == "__main__":
    main()
