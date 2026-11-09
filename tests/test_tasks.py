# -*- coding: utf-8 -*-
"""Tests for the task registry."""
from __future__ import annotations

import unittest

from echoforge.tasks import TASKS, list_tasks, get_task


class TestRegistry(unittest.TestCase):
    def test_list(self):
        self.assertEqual(list_tasks(), TASKS)

    def test_ids(self):
        ids = {t["id"] for t in TASKS}
        self.assertIn("retrieve", ids)
        self.assertIn("route", ids)
        self.assertIn("echo", ids)

    def test_get(self):
        self.assertEqual(get_task("retrieve")["name"], "长上下文检索")

    def test_get_missing(self):
        self.assertIsNone(get_task("nope"))


if __name__ == "__main__":
    unittest.main()
