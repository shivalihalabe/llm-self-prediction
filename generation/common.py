#!/usr/bin/env python3
"""
Shared maze and prompt helpers for the generation scripts
---

Defines the maze-topology description, the prediction user message and the answer parser
once, so every generation script builds byte-identical prompts and parses answers the same
way.

Notes:
- these helpers are part of the experimental protocol, not conveniences
- test_helpers.py pins their outputs against fixtures captured from the collected data
"""

import json
import re

ROWS, COLS = 5, 5


def parse_walls(wl):
    """Wall list [[a, b], ...] from a navigation file -> set of frozenset cell pairs."""
    return set(frozenset([tuple(p[0]), tuple(p[1])]) for p in wl)


def get_available_directions(pos, walls, rows=ROWS, cols=COLS):
    """Return dict: direction_name -> neighbor_cell for valid moves from pos."""
    r, c = pos
    candidates = {
        "North": (r - 1, c),
        "South": (r + 1, c),
        "East":  (r, c + 1),
        "West":  (r, c - 1),
    }
    out = {}
    for d, (nr, nc) in candidates.items():
        if 0 <= nr < rows and 0 <= nc < cols and frozenset([pos, (nr, nc)]) not in walls:
            out[d] = (nr, nc)
    return out


def describe_maze_topology(walls):
    """Full topology block shown to the model: every cell and its available directions."""
    lines = [f"Grid maze: {ROWS}x{COLS}. Positions (row,col), (0,0) top-left.",
             "Directions from each position:"]
    for r in range(ROWS):
        for c in range(COLS):
            dirs = get_available_directions((r, c), walls)
            if dirs:
                pairs = ", ".join(f"{n}->({a},{b})" for n, (a, b) in sorted(dirs.items()))
                lines.append(f"  ({r},{c}): {pairs}")
    return "\n".join(lines)


def build_user_msg(walls, n_steps):
    """The prediction user message: topology plus the step question."""
    return (
        f"{describe_maze_topology(walls)}\n\n"
        f"Starting at (0, 0), predict the position after {n_steps} steps."
    )


def parse_answer(content, reasoning=True):
    """Parse a predicted position from a response; None if unparseable.

    With reasoning=False the answer was requested as JSON {"row", "col"}, so JSON is tried
    first; both modes fall back to the last "(row, col)" match in the text.
    """
    if not content:
        return None
    if not reasoning:
        try:
            o = json.loads(content)
            r, c = int(o["row"]), int(o["col"])
            if 0 <= r < ROWS and 0 <= c < COLS:
                return [r, c]
        except Exception:
            pass
    ms = re.findall(r"\((\d+)\s*,\s*(\d+)\)", content)
    if ms:
        r, c = int(ms[-1][0]), int(ms[-1][1])
        if 0 <= r < ROWS and 0 <= c < COLS:
            return [r, c]
    return None
