# -*- coding: utf-8 -*-
"""Tests for the CLI dispatch (smoke: commands run without raising)."""
from __future__ import annotations

import unittest

from echoforge.cli import main


class TestCLI(unittest.TestCase):
    def test_doctor(self):
        main(["doctor"])

    def test_bench(self):
        main(["bench"])

    def test_modes(self):
        main(["modes"])

    def test_retrieve(self):
        main(["retrieve", "--skill", "0.5"])

    def test_route(self):
        main(["route", "--token", "10", "--skill", "0.5"])

    def test_echo(self):
        main(["echo", "--token", "10", "--skill", "0.5"])

    def test_train(self):
        main(["train", "--steps", "3"])


if __name__ == "__main__":
    unittest.main()
