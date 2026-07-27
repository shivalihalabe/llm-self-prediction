#!/usr/bin/env python3
"""
Maze-structural drivers of predictability
---

Asks which maze features make a maze predictable, net of which model is navigating.
Predictability per maze is the model-centered self-accuracy.

Method:
- take each model's per-maze accuracy minus its own overall self-accuracy, then average over
  the models consistent on that maze
- centering removes the per-model level, so the residual reflects the maze
- correlate that maze effect (Pearson and permutation p) with BFS depth, junction count, and
  the mean number of branch decisions the models faced

Measures:
- per-maze predictability and features, on two maze sets
- whether a self-predictable maze is also cross-predictable
- where decision points sit in the run
- whether the consistent sets differ in structural difficulty

Output: analysis/results/maze_structure.json
"""

import json
import os
import statistics as st

import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("maze_structure")}


# n_walls, mean_open_degree and n_reachable are constant across all 100 mazes (10 walls,
# degree 2.4, 25 reachable), so they stay in the per_maze dump but not in the correlations.
FEATURES = [
    "bfs_max_depth",
    "n_junctions",
    "mean_branch_steps",
]


def maze_features(mz):
    """Structural features of a maze, computed from its walls alone."""
    d = C.bfs_dist(mz)
    degs, junctions = [], 0
    for c in d:
        od = sum(1 for nb in C.neighbors(c) if frozenset([c, nb]) not in C.WALLS[mz])
        degs.append(od)
        if od >= 3:
            junctions += 1
    return {
        "n_walls": len(C.WALLS[mz]),
        "n_reachable": len(d),
        "bfs_max_depth": max(d.values()),
        "mean_open_degree": round(st.mean(degs), 3),
        "n_junctions": junctions,
    }


# Per-maze predictability + features

overall = {m: C.acc(C.SELF[m])[0] for m in MODELS}
per_maze = {}
for mz in sorted(set().union(*C.CONSISTENT.values())):
    raw, centered, branchy = [], [], []
    for m in MODELS:
        if mz not in C.CONSISTENT[m]:
            continue
        a = C.acc(C.SELF[m], {mz})[0]
        if a is None:
            continue
        raw.append(a)
        centered.append(a - overall[m])
        branchy.append(sum(1 for s in range(1, len(C.TRUTH[m][mz])) if C.is_branch(m, mz, s)))
    if not centered:
        continue
    per_maze[mz] = {
        **maze_features(mz),
        "n_models": len(centered),
        "mean_self_acc": round(st.mean(raw), 1),
        "maze_effect": round(st.mean(centered), 1),  # + => easier than the navigator's average
        "mean_branch_steps": round(st.mean(branchy), 2),
    }
RES["per_maze"] = per_maze

_pm = pd.DataFrame.from_dict(per_maze, orient="index")


# Correlations (two maze sets)

def correlate(mazes):
    """Pearson r and permutation p for each feature against the maze effect."""
    sub = _pm.loc[mazes]
    eff = sub["maze_effect"].tolist()
    out = {"n": len(mazes)}
    for feat in FEATURES:
        xs = sub[feat].tolist()
        out[feat] = {"pearson": C.pearson(xs, eff), "perm_p": C.perm_corr_p(xs, eff)}
    return out


inter = sorted(mz for mz in C.INTERSECTION if mz in per_maze)
k3 = sorted(_pm.index[_pm.n_models >= 3])
RES["correlations_intersection19"] = correlate(inter)
RES["correlations_k3plus"] = correlate(k3)


# Self-predictable vs cross-predictable mazes
# Per maze (on the 5-way intersection): mean self-accuracy across models vs mean
# cross-accuracy across pairs.

self_cross = []
for mz in inter:
    self_vals = [C.acc(C.SELF[m], {mz})[0] for m in MODELS if mz in C.CONSISTENT[m]]
    self_vals = [v for v in self_vals if v is not None]
    cross_vals = [
        C.acc(C.CROSS[(p, t)], {mz})[0]
        for p in MODELS
        for t in MODELS
        if p != t and (p, t) in C.CROSS
    ]
    cross_vals = [v for v in cross_vals if v is not None]
    if self_vals and cross_vals:
        self_cross.append((st.mean(self_vals), st.mean(cross_vals)))
RES["self_vs_cross_predictability_per_maze"] = {
    "n_mazes": len(self_cross),
    "pearson": C.pearson([x for x, _ in self_cross], [y for _, y in self_cross]),
    "perm_p": C.perm_corr_p([x for x, _ in self_cross], [y for _, y in self_cross]),
}


# Decision-point position in the run
# Complements the parity check above: junction counts say how branchy the mazes are, not where
# the branches sit relative to the start. Unlike junctions and BFS depth this is run-dependent
# (it uses the routes taken), which is appropriate here: the question is whether the models
# encountered choices at different points in the run, not whether the mazes could
# have produced that.

position = {}
for m in MODELS:
    steps, firsts = [], []
    for mz in sorted(C.CONSISTENT[m]):
        dps = [s2 for s2 in range(1, len(C.TRUTH[m][mz])) if C.is_branch(m, mz, s2)]
        steps += dps
        if dps:
            firsts.append(dps[0])
    position[m] = {
        "decision_points_by_step_pct": {
            s2: round(100.0 * sum(1 for x in steps if x == s2) / len(steps)) for s2 in range(1, 9)
        },
        "mean_step": round(st.mean(steps), 2),
        "median_step": float(st.median(steps)),
        "first_decision_point": {
            "mean": round(st.mean(firsts), 2),
            "median": float(st.median(firsts)),
            "n_mazes_with_decision_point": len(firsts),
        },
    }
RES["decision_point_position"] = position


# Per-maze self vs cross pairs
# The per-maze points behind the self-vs-cross correlation above: mean self-accuracy across the
# models consistent on the maze vs mean cross-accuracy across all ordered pairs, intersection set.

RES["per_maze_self_vs_cross"] = [
    {"maze": mz, "self": x, "cross": y} for mz, (x, y) in zip(inter, self_cross)
]


# Consistent-set structural difficulty
# Whether models differ in the structural difficulty of the mazes they navigate consistently.
# Structural measures only, computed from walls. Route-dependent features (mean_branch_steps)
# and navigator-centred ones (maze_effect) are excluded here; both remain in per_maze.

cons_diff = {}
for m in MODELS:
    mzs = sorted(C.CONSISTENT[m])
    cons_diff[m] = {
        "n_mazes": len(mzs),
        "mean_n_junctions": round(st.mean(per_maze[mz]["n_junctions"] for mz in mzs), 2),
        "mean_bfs_depth": round(st.mean(per_maze[mz]["bfs_max_depth"] for mz in mzs), 2),
    }
cons_diff["all_mazes"] = {
    "n_mazes": len(per_maze),
    "mean_n_junctions": round(st.mean(d["n_junctions"] for d in per_maze.values()), 2),
    "mean_bfs_depth": round(st.mean(d["bfs_max_depth"] for d in per_maze.values()), 2),
}
RES["consistent_set_difficulty"] = cons_diff


# Write

with open(os.path.join(OUT, "maze_structure.json"), "w") as f:
    json.dump(RES, f, indent=1)
