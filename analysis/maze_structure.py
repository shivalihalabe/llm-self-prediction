#!/usr/bin/env python3
"""
Maze-structural drivers of predictability (the maze-side complement)
====================================================================

The model-side analysis showed predictability is largely a model property (rule-like branch
choices). This asks the complementary question: net of which model is navigating, what makes a
*maze* easy or hard to predict?

Predictability per maze is the model-CENTERED self-accuracy: each model's per-maze accuracy
minus its own overall self-accuracy, averaged over the models consistent on that maze. Centering
removes the dominant "Qwen always high / Sonnet always low" effect so the residual reflects the
maze, not the navigator. That maze effect is correlated (Pearson + permutation p) with intrinsic
features (interior walls, reachable area, BFS depth, branchiness, junctions) and a behavioral
feature (mean number of genuine branch decisions the models faced on the maze).

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

FEATURES = [
    "n_walls",
    "n_reachable",
    "bfs_max_depth",
    "mean_open_degree",
    "n_junctions",
    "mean_branch_steps",
]


def maze_features(mz):
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


# ============================================================
# PER-MAZE PREDICTABILITY + FEATURES
# ============================================================

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


# ============================================================
# CORRELATIONS (two maze sets)
# ============================================================


def correlate(mazes):
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


# ============================================================
# IS A SELF-PREDICTABLE MAZE ALSO CROSS-PREDICTABLE?
# ============================================================

# Per maze (on the 5-way intersection): mean self-accuracy across models vs mean cross-accuracy across pairs.
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
    "note": "high => the same mazes are easy/hard whether a model predicts itself or another model predicts it",
}


# ============================================================
# PER-MAZE SELF VS CROSS PAIRS
# ============================================================

# The per-maze points behind the self-vs-cross correlation above: mean self-accuracy across the
# models consistent on the maze vs mean cross-accuracy across all ordered pairs, intersection set.
RES["per_maze_self_vs_cross"] = [
    {"maze": mz, "self": x, "cross": y} for mz, (x, y) in zip(inter, self_cross)
]


# ============================================================
# CONSISTENT-SET STRUCTURAL DIFFICULTY
# ============================================================

# Whether models differ in the structural difficulty of the mazes they navigate consistently.
# Uses the model-independent structural measures (branch steps, BFS depth), not maze_effect,
# which is centred per navigator and therefore circular for this comparison. The all-mazes row
# is the baseline: a consistent set harder or easier than it reflects selection.
cons_diff = {}
for m in MODELS:
    mzs = sorted(C.CONSISTENT[m])
    cons_diff[m] = {
        "n_mazes": len(mzs),
        "mean_branch_steps": round(st.mean(per_maze[mz]["mean_branch_steps"] for mz in mzs), 2),
        "mean_bfs_depth": round(st.mean(per_maze[mz]["bfs_max_depth"] for mz in mzs), 2),
    }
cons_diff["all_mazes"] = {
    "n_mazes": len(per_maze),
    "mean_branch_steps": round(st.mean(d["mean_branch_steps"] for d in per_maze.values()), 2),
    "mean_bfs_depth": round(st.mean(d["bfs_max_depth"] for d in per_maze.values()), 2),
}
RES["consistent_set_difficulty"] = cons_diff


# ============================================================
# WRITE + SUMMARY
# ============================================================

with open(os.path.join(OUT, "maze_structure.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print(
        f"per-maze predictability computed for {len(per_maze)} mazes "
        f"(model-centered maze effect; + = easier than navigator's own average)."
    )
    print("\nmaze effect vs structural features:")
    print(f"  {'feature':16} {'k>=3 (n=' + str(len(k3)) + ')':>22} {'intersection19':>22}")
    for feat in FEATURES:
        a = RES["correlations_k3plus"][feat]
        b = RES["correlations_intersection19"][feat]
        print(
            f"  {feat:16} r={str(a['pearson']):>7} (p={str(a['perm_p']):>6})   r={str(b['pearson']):>7} (p={str(b['perm_p']):>6})"
        )
    ranked = sorted(k3, key=lambda mz: per_maze[mz]["maze_effect"])
    print("\nhardest (most negative maze effect):")
    for mz in ranked[:3]:
        d = per_maze[mz]
        print(
            f"  {mz:8} effect={d['maze_effect']:+.1f}  walls={d['n_walls']} junctions={d['n_junctions']} branch_steps={d['mean_branch_steps']} depth={d['bfs_max_depth']}"
        )
    print("easiest (most positive maze effect):")
    for mz in ranked[-3:]:
        d = per_maze[mz]
        print(
            f"  {mz:8} effect={d['maze_effect']:+.1f}  walls={d['n_walls']} junctions={d['n_junctions']} branch_steps={d['mean_branch_steps']} depth={d['bfs_max_depth']}"
        )
    sc = RES["self_vs_cross_predictability_per_maze"]
    print(
        f"\nself vs cross predictability per maze: r={sc['pearson']} (perm p={sc['perm_p']}, n={sc['n_mazes']})"
    )
    print("\n-> wrote results/maze_structure.json")
