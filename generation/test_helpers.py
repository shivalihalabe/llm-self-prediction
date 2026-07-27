#!/usr/bin/env python3
"""
Fixture test for the shared generation helpers
==============================================

The expected values in test_fixtures.json were captured from the per-script helper copies
before they were consolidated into common.py, on mazes from data/navigation and raw
responses from the committed prediction files (at least one per model, both answer modes).
A refactor that changes prompt construction or answer parsing fails here rather than
surfacing only if data were ever re-collected.

Run: python3 generation/test_helpers.py
"""

import json
import os

from common import build_user_msg, describe_maze_topology, get_available_directions, parse_walls
from common import parse_answer

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = json.load(open(os.path.join(_HERE, "test_fixtures.json")))
NAV = json.load(open(os.path.join(_HERE, "..", "data", "navigation", "opus_navigation.json")))
WALLS_RAW = {m["id"]: m["walls"] for m in NAV["mazes"]}


def test_describe_maze_topology():
    for mid, expected in FIXTURES["describe_maze_topology"].items():
        assert describe_maze_topology(parse_walls(WALLS_RAW[mid])) == expected, mid


def test_get_available_directions():
    for mid, cells in FIXTURES["get_available_directions"].items():
        walls = parse_walls(WALLS_RAW[mid])
        for key, expected in cells.items():
            r, c = map(int, key.split(","))
            got = get_available_directions((r, c), walls)
            assert {k: list(v) for k, v in got.items()} == expected, (mid, key)
            assert list(got) == list(expected), (mid, key, "direction order changed")


def test_build_user_msg():
    for (mid, expected), n_steps in zip(FIXTURES["build_user_msg"].items(), (1, 4, 8)):
        assert build_user_msg(parse_walls(WALLS_RAW[mid]), n_steps) == expected, mid


def test_parse_answer():
    for s in FIXTURES["parse_answer"]:
        got = parse_answer(s["raw"], reasoning=s["mode"] == "reasoning")
        assert got == s["expected"], (s["mode"], s["raw"][:60], got, s["expected"])


if __name__ == "__main__":
    for fn in (test_describe_maze_topology, test_get_available_directions,
               test_build_user_msg, test_parse_answer):
        fn()
        print(f"{fn.__name__}: ok")
    print("all fixture tests passed")
