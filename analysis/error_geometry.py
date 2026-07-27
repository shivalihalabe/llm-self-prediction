#!/usr/bin/env python3
"""
Error geometry
==============
When a self-prediction is wrong, *how* is it wrong.

Goes beyond binary exact-match using the predicted coordinate
(common.SELF_POS) vs the true trajectory: Manhattan distance, off-by-k-step errors
(right path / wrong count), on-path vs off-path, over/under-shoot, geometric reachability,
marginal row/column accuracy, accuracy stratified by the true endpoint, and the step-8
corner attractor. Computed for self-prediction (each model predicting itself).

Output: analysis/results/error_geometry.json
"""

import json
import os
import collections

import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("error_geometry")}


def _offset(t, maze, s, pred):
    """Signed step-offset for an on-path prediction (None if off-path): predicted index - true step."""
    idxs = [j for j, p in enumerate(C.TRUTH[t][maze]) if list(p) == list(pred)]
    if not idxs:
        return None
    return min(idxs, key=lambda j: abs(j - s)) - s


def _self_frame(t):
    """One row per scored self-prediction with its geometric annotations."""
    rows = []
    for (maze, s), pred in C.SELF_POS[t].items():
        truth = list(C.TRUTH[t][maze][s])
        rows.append(
            (
                maze,
                s,
                pred[0],
                pred[1],
                truth[0],
                truth[1],
                C.manhattan(pred, truth),
                tuple(pred) in C.reachable_exactly(maze, s),
                _offset(t, maze, s, pred),
            )
        )
    return pd.DataFrame(
        rows, columns=["maze", "step", "pr", "pc", "tr", "tc", "dist", "reachable", "offset"]
    )


FRAMES = {t: _self_frame(t) for t in MODELS}


# ============================================================
# PER-TARGET GEOMETRY OF ERRORS
# ============================================================

geom = {}
for t in MODELS:
    df = FRAMES[t]
    n = len(df)
    wrong = df[df.dist > 0]
    on_path_wrong = wrong[wrong.offset.notna()]
    offsets = collections.Counter(int(o) for o in on_path_wrong.offset)
    geom[t] = {
        "n": n,
        "exact_acc": C.pct((df.dist == 0).mean()) if n else None,
        "mean_dist_all": round(float(df.dist.mean()), 2) if n else None,
        "mean_dist_when_wrong": round(float(wrong.dist.mean()), 2) if len(wrong) else None,
        "row_acc": C.pct((df.pr == df.tr).mean()) if n else None,
        "col_acc": C.pct((df.pc == df.tc).mean()) if n else None,
        "frac_reachable": C.pct(df.reachable.mean()) if n else None,
        "of_wrong_frac_on_path": C.pct(len(on_path_wrong), len(wrong)),
        "of_onpath_wrong_overshoot": int((on_path_wrong.offset > 0).sum()),
        "of_onpath_wrong_undershoot": int((on_path_wrong.offset < 0).sum()),
        "step_offset_hist": {k: v for k, v in sorted(offsets.items())},
    }
RES["self_error_geometry"] = geom


# ============================================================
# ACCURACY STRATIFIED BY TRUE ENDPOINT ROW
# ============================================================

# Does the model do better when the truth sits where its prior expects (e.g. top rows)?
endpoint = {}
for t in MODELS:
    df = FRAMES[t]
    endpoint[t] = {}
    for r in range(C.ROWS):
        sub = df[df.tr == r]
        endpoint[t][r] = {
            "acc": C.pct((sub.dist == 0).mean()) if len(sub) else None,
            "n": int(len(sub)),
        }
RES["accuracy_by_true_row"] = endpoint


# ============================================================
# STEP-8 CORNER ATTRACTOR
# ============================================================

corner = {}
for t in MODELS:
    mazes = [mz for mz in sorted(C.CONSISTENT[t]) if 8 < len(C.TRUTH[t][mz])]
    actual_44 = sum(1 for mz in mazes if tuple(C.TRUTH[t][mz][8]) == (4, 4))
    pred_44 = sum(
        1 for mz in mazes if (mz, 8) in C.SELF_POS[t] and tuple(C.SELF_POS[t][(mz, 8)]) == (4, 4)
    )
    pred_dist = collections.Counter(
        tuple(C.SELF_POS[t][(mz, 8)]) for mz in mazes if (mz, 8) in C.SELF_POS[t]
    )
    corner[t] = {
        "n_mazes_with_step8": len(mazes),
        "actual_end_at_4_4": actual_44,
        "predicted_4_4": pred_44,
        "top3_predicted_step8": [[list(p), c] for p, c in pred_dist.most_common(3)],
    }
RES["corner_attractor_step8"] = corner


# ============================================================
# PREDICTED VS ACTUAL POSITION DISTRIBUTION
# ============================================================

# Are predictions systematically biased toward certain cells (beyond the step-8 corner)?
dist_div = {}
for t in MODELS:
    per_step, pred_tot, act_tot = [], collections.Counter(), collections.Counter()
    for s in range(1, 9):
        items = [
            (mz, tuple(C.SELF_POS[t][(mz, s)]))
            for mz in sorted(C.CONSISTENT[t])
            if (mz, s) in C.SELF_POS[t]
        ]
        if not items:
            per_step.append(None)
            continue
        pc = collections.Counter(p for _, p in items)
        ac = collections.Counter(tuple(C.TRUTH[t][mz][s]) for mz, _ in items)
        n = len(items)
        tv = 0.5 * sum(abs(pc[c] / n - ac[c] / n) for c in set(pc) | set(ac))
        per_step.append(round(tv, 3))
        pred_tot += pc
        act_tot += ac
    cells = sorted(set(pred_tot) | set(act_tot))  # sorted: deterministic tie order for max/min
    diff = {c: pred_tot[c] - act_tot[c] for c in cells}
    over, under = max(diff, key=diff.get), min(diff, key=diff.get)
    vals = [v for v in per_step if v is not None]
    dist_div[t] = {
        "mean_tv": round(sum(vals) / len(vals), 3) if vals else None,
        "per_step_tv": per_step,
        "most_overpredicted_cell": [list(over), diff[over]],
        "most_underpredicted_cell": [list(under), diff[under]],
    }
RES["prediction_distribution_divergence"] = dist_div


# ============================================================
# ROW / COLUMN CONFUSION (actual -> predicted)
# ============================================================

row_conf, col_conf = {}, {}
for t in MODELS:
    df = FRAMES[t]
    rc = pd.crosstab(df.tr, df.pr).reindex(index=range(C.ROWS), columns=range(C.ROWS), fill_value=0)
    cc = pd.crosstab(df.tc, df.pc).reindex(index=range(C.COLS), columns=range(C.COLS), fill_value=0)
    row_conf[t] = rc.values.tolist()
    col_conf[t] = cc.values.tolist()
RES["row_confusion_actual_to_predicted"] = row_conf
RES["col_confusion_actual_to_predicted"] = col_conf


# ============================================================
# ERROR GEOMETRY BY STEP (horizon-resolved)
# ============================================================

geo_step = {}
for t in MODELS:
    df = FRAMES[t]
    rows = []
    for s in range(1, 9):
        sub = df[df.step == s]
        w = sub[sub.dist > 0]
        rows.append(
            {
                "step": s,
                "n": int(len(sub)),
                "row_acc": C.pct((sub.pr == sub.tr).mean()) if len(sub) else None,
                "mean_dist_when_wrong": round(float(w.dist.mean()), 2) if len(w) else None,
                "overshoot": int((w.offset > 0).sum()),
                "undershoot": int((w.offset < 0).sum()),
            }
        )
    geo_step[t] = rows
RES["geometry_by_step"] = geo_step


# ============================================================
# WRITE
# ============================================================

with open(os.path.join(OUT, "error_geometry.json"), "w") as f:
    json.dump(RES, f, indent=1)
