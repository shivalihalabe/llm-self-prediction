#!/usr/bin/env python3
"""
Outcome analyses
================
Who predicts whom, and how well.

Covers the predictor x target accuracy matrix (native + 5-way-intersection), self-vs-other gaps,
per-step horizon decay, target-predictability vs predictor-skill, pairwise comparisons,
self-accuracy by maze difficulty, and the reasoning vs no-reasoning self contrast.

All accuracy is the shared scoring contract from common.py (run_idx 0, exact match).
Output: analysis/results/outcomes.json
"""

import json
import os
import statistics as st
import itertools

import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

MODELS = C.MODELS
RES = {"metadata": C.metadata("outcomes")}

# per-(predictor, target[, step]) accuracy tables, straight off the tidy record frame
_PT = C.RECORDS[C.RECORDS.kind.isin(("self", "cross"))]


def _acc_table(df, by):
    """% accuracy grouped by the given columns."""
    return df.groupby(by, sort=False)["correct"].mean().mul(100.0)


def a(scored, mazeset=None, step=None):
    """accuracy as a rounded float or None."""
    v = C.acc(scored, mazeset, step)[0]
    return round(v, 1) if v is not None else None


# ============================================================
# MATRIX
# ============================================================


def matrix(mazeset):
    tab = C.accuracy_matrix(mazeset=mazeset).round(1)
    return {
        p: {t: (None if pd.isna(tab.loc[p, t]) else float(tab.loc[p, t])) for t in MODELS}
        for p in MODELS
    }


RES["matrix_native"] = matrix(None)  # each cell on the target's consistent set
RES["matrix_intersection19"] = matrix(C.INTERSECTION)  # all cells on the same 19 mazes


# ============================================================
# SELF VS OTHER
# ============================================================


def self_vs_other(mazeset):
    df = _PT if mazeset is None else _PT[_PT.maze.isin(mazeset)]
    acc = _acc_table(df, ["predictor", "target"])
    out = {}
    for t in MODELS:
        if (t, t) not in acc.index:
            continue
        self_a = float(acc[(t, t)])
        others = {p: float(acc[(p, t)]) for p in MODELS if p != t and (p, t) in acc.index}
        if not others:
            continue
        out[t] = {
            "self": round(self_a, 1),
            "mean_other": round(st.mean(others.values()), 1),
            "median_other": round(st.median(others.values()), 1),
            "best_other": round(max(others.values()), 1),
            "best_other_model": max(others, key=others.get),
            "gap_vs_mean": round(self_a - st.mean(others.values()), 1),
            "gap_vs_best": round(self_a - max(others.values()), 1),
            "per_other": {p: round(v, 1) for p, v in others.items()},
            "n_self": int(((df.predictor == t) & (df.target == t)).sum()),
        }
    return out


RES["self_vs_other_native"] = self_vs_other(None)
RES["self_vs_other_intersection19"] = self_vs_other(C.INTERSECTION)


# ============================================================
# PER-STEP SELF VS OTHER
# ============================================================

_step_acc = _acc_table(_PT, ["target", "predictor", "step"])
_step_n = _PT.groupby(["target", "predictor", "step"], sort=False)["correct"].size()

perstep = {}
for t in MODELS:
    rows = {k: [] for k in ("self", "mean_other", "best_other", "gap_vs_mean", "gap_vs_best", "n")}
    for s in range(1, 9):
        self_a = _step_acc.get((t, t, s))
        self_a = float(self_a) if self_a is not None else None
        others = [
            float(_step_acc[(t, p, s)]) for p in MODELS if p != t and (t, p, s) in _step_acc.index
        ]
        if self_a is None or not others:
            for k in rows:
                rows[k].append(None)
            continue
        rows["self"].append(round(self_a))
        rows["mean_other"].append(round(st.mean(others)))
        rows["best_other"].append(round(max(others)))
        rows["gap_vs_mean"].append(round(self_a - st.mean(others)))
        rows["gap_vs_best"].append(round(self_a - max(others)))
        rows["n"].append(int(_step_n[(t, t, s)]))
    perstep[t] = rows
RES["per_step_self_vs_other"] = perstep


# ============================================================
# PREDICTABILITY / SKILL PER STEP
# ============================================================

# mean of per-predictor accuracies (not pooled), matching the per-predictor cell definition
RES["target_predictability_per_step"] = {
    t: [
        int(round(float(v))) if pd.notna(v) else None
        for v in _step_acc.xs(t, level="target").groupby("step").mean().reindex(range(1, 9))
    ]
    for t in MODELS
}

_cross_step = _acc_table(_PT[_PT.predictor != _PT.target], ["predictor", "target", "step"])
RES["predictor_skill_per_step"] = {
    p: [
        int(round(float(v))) if pd.notna(v) else None
        for v in _cross_step.xs(p, level="predictor").groupby("step").mean().reindex(range(1, 9))
    ]
    for p in MODELS
}


# ============================================================
# PAIRWISE (on pairwise intersection)
# ============================================================

pairwise = {}
for a_m, b_m in itertools.combinations(MODELS, 2):
    s = C.PAIRWISE[tuple(sorted((a_m, b_m)))]
    pairwise[f"{a_m}|{b_m}"] = {
        "n_mazes": len(s),
        f"{a_m}_self": a(C.SELF[a_m], s),
        f"{b_m}_self": a(C.SELF[b_m], s),
        f"{a_m}->{b_m}": a(C.CROSS[(a_m, b_m)], s) if (a_m, b_m) in C.CROSS else None,
        f"{b_m}->{a_m}": a(C.CROSS[(b_m, a_m)], s) if (b_m, a_m) in C.CROSS else None,
    }
RES["pairwise_on_intersection"] = pairwise


# ============================================================
# SELF ACCURACY BY MAZE DIFFICULTY
# ============================================================

# difficulty stratum k = mazes consistent for exactly k models.
diff = {}
for k in range(1, 6):
    strat = C.DIFFICULTY_STRATA[k]
    per_model = {
        m: a(C.SELF[m], strat) for m in MODELS
    }  # None where a model has no consistent maze in stratum
    vals = [v for v in per_model.values() if v is not None]
    diff[k] = {
        "n_mazes": len(strat),
        "self_acc_per_model": per_model,
        "mean_self_acc": round(st.mean(vals), 1) if vals else None,
    }
RES["self_accuracy_by_difficulty"] = diff


# ============================================================
# REASONING VS NO-REASONING (self)
# ============================================================

rvn = {}
for m in MODELS:
    common_keys = set(C.SELF[m]) & set(C.SELF_NR[m])
    rvn[m] = {
        "reasoning": a(C.SELF[m]),
        "noreasoning": a(C.SELF_NR[m]),
        "gap": round(C.acc(C.SELF[m])[0] - C.acc(C.SELF_NR[m])[0], 1),
        "reasoning_on_common": a({k: C.SELF[m][k] for k in common_keys}),
        "nr_on_common": a({k: C.SELF_NR[m][k] for k in common_keys}),
        "per_step_reasoning": [a(C.SELF[m], None, s) for s in range(1, 9)],
        "per_step_nr": [a(C.SELF_NR[m], None, s) for s in range(1, 9)],
    }
RES["reasoning_vs_nr_self"] = rvn


# ============================================================
# WRITE + CONSOLE SUMMARY
# ============================================================

with open(os.path.join(OUT, "outcomes.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print("matrix (native), predictor rows -> target cols:")
    print("        " + "".join(f"{t:>8}" for t in MODELS))
    for p in MODELS:
        print(f"  {p:6}" + "".join(f"{str(RES['matrix_native'][p][t]):>8}" for t in MODELS))
    print("\nself vs best-other (native): gap_vs_best")
    for t, d in RES["self_vs_other_native"].items():
        print(
            f"  {t:7} self={d['self']:5}  best_other={d['best_other']:5} ({d['best_other_model']:6})  gap={d['gap_vs_best']:+.1f}  n={d['n_self']}"
        )
    print("\nself accuracy by difficulty (mean over models):")
    for k in range(1, 6):
        print(
            f"  {k} models consistent ({RES['self_accuracy_by_difficulty'][k]['n_mazes']:2} mazes): {RES['self_accuracy_by_difficulty'][k]['mean_self_acc']}"
        )
    print("\nreasoning vs no-reasoning (self):")
    for m, d in rvn.items():
        print(f"  {m:7} R={d['reasoning']:5}  NR={d['noreasoning']:5}  gap={d['gap']:+.1f}")
    print("\n-> wrote results/outcomes.json")
