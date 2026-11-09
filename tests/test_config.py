# -*- coding: utf-8 -*-
"""Tests for config loading and the yaml subset parser."""
from __future__ import annotations

import os
import tempfile
import unittest

from echoforge.config import load_config, DEFAULTS, _merge
from echoforge.utils.yamlish import yaml_loads, yaml_dump


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config()
        self.assertEqual(cfg["attention"]["vocab"], 256)
        self.assertEqual(cfg["moe"]["n_experts"], 64)
        self.assertEqual(cfg["echo"]["k_steps"], 4)

    def test_recipe_merge(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write("attention:\n  window: 4\nmoe:\n  top_k: 2\n")
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["attention"]["window"], 4)
            self.assertEqual(cfg["moe"]["top_k"], 2)
            self.assertEqual(cfg["attention"]["seq_len"], 64)  # untouched default
        finally:
            os.unlink(path)

    def test_merge_nested(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        self.assertEqual(_merge(base, {"a": {"x": 9}})["a"], {"x": 9, "y": 2})


class TestYamlish(unittest.TestCase):
    def test_scalars(self):
        d = yaml_loads("a: 1\nb: 2.5\nc: true\nd: hello\n")
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2.5)
        self.assertIs(d["c"], True)
        self.assertEqual(d["d"], "hello")

    def test_nested(self):
        d = yaml_loads("a:\n  b: 1\n  c: 2\n")
        self.assertEqual(d["a"]["b"], 1)

    def test_block_list(self):
        d = yaml_loads("key:\n  - a\n  - b\n")
        self.assertEqual(d["key"], ["a", "b"])

    def test_two_lists(self):
        d = yaml_loads("x:\n  - 1\n  - 2\ny:\n  - 3\n")
        self.assertEqual(d["x"], [1, 2])
        self.assertEqual(d["y"], [3])

    def test_roundtrip(self):
        d = yaml_loads("a: 1\nb:\n  - x\n  - y\n")
        d2 = yaml_loads(yaml_dump(d))
        self.assertEqual(d, d2)


if __name__ == "__main__":
    unittest.main()
