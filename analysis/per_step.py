#!/usr/bin/env python3
"""
Horizon-resolved analyses (the per-step lens)
=============================================

Consolidates and extends step-by-step views that don't belong to a single thematic script:
- self_consistency_by_step: where over the horizon do a model's runs stop agreeing (is Opus's
  temp-0 instability concentrated mid-horizon, exactly where its self-advantage lives?).
- forced_vs_branch_by_step: self-accuracy on forced vs branch steps, per step.
- self_matches_consensus_by_step: does self track the consensus-of-others more as the horizon grows?
- error_propagation: P(correct at k+1 | correct at k) vs P(correct at k+1 | wrong at k) -- are
  steps independent or does getting one right carry forward.
- predictability_horizon: the first step at which each target drops below 50% mean predictability.

Output: analysis/results/per_step.json
"""

import json
import os
import collections

import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("per_step")}


# ============================================================
# SELF-CONSISTENCY BY STEP
# ============================================================


def _consistency(m):
    rows = [
        (mz, s, ri, tuple(rec["parsed_position"]))
        for mz, s, ri, rec in C.self_records(m, "reasoning")
        if rec.get("parsed_position") is not None
    ]
    df = pd.DataFrame(rows, columns=["maze", "step", "run", "pos"])
    df = df[df.step.between(1, 8)]
    g = df.groupby(["maze", "step"])["pos"].agg(n_runs="size", n_distinct="nunique").reset_index()
    g = g[g.n_runs >= 2]
    per = g.assign(agree=g.n_distinct == 1).groupby("step")["agree"].agg(["mean", "size"])
    return {
        s: {
            "agree_frac": round(float(per.loc[s, "mean"]), 3) if s in per.index else None,
            "n": int(per.loc[s, "size"]) if s in per.index else 0,
        }
        for s in range(1, 9)
    }


RES["self_consistency_by_step"] = {m: _consistency(m) for m in MODELS}


# ============================================================
# FORCED VS BRANCH ACCURACY BY STEP
# ============================================================


def _forced_vs_branch(m):
    df = pd.DataFrame(
        [(mz, s, c, len(C.unvisited_moves(m, mz, s))) for (mz, s), c in C.SELF[m].items()],
        columns=["maze", "step", "correct", "n_unvisited"],
    )
    rows = []
    for s in range(1, 9):
        f = df[(df.step == s) & (df.n_unvisited == 1)]["correct"]
        b = df[(df.step == s) & (df.n_unvisited >= 2)]["correct"]
        rows.append(
            {
                "step": s,
                "forced_acc": C.pct(f.mean()) if len(f) else None,
                "n_forced": int(len(f)),
                "branch_acc": C.pct(b.mean()) if len(b) else None,
                "n_branch": int(len(b)),
            }
        )
    return rows


RES["forced_vs_branch_by_step"] = {m: _forced_vs_branch(m) for m in MODELS}


# ============================================================
# SELF MATCHES CONSENSUS-OF-OTHERS BY STEP
# ============================================================

_ORDER = {p: i for i, p in enumerate(MODELS)}


def _consensus(votes):
    """Modal position; ties break by predictor order (Counter insertion order)."""
    return collections.Counter(votes).most_common(1)[0][0]


def consensus_by_step(t):
    cross = C.RECORDS[(C.RECORDS.kind == "cross") & (C.RECORDS.target == t)].copy()
    cross["ord"] = cross.predictor.map(_ORDER)
    cons = cross.sort_values("ord").groupby(["maze", "step"])["pred"].agg(_consensus)
    self_df = C.RECORDS[(C.RECORDS.kind == "self") & (C.RECORDS.target == t)]
    merged = self_df.merge(cons.rename("consensus"), on=["maze", "step"], how="inner")
    rows = []
    for s in range(1, 9):
        sub = merged[merged.step == s]
        n = len(sub)
        rows.append(
            {
                "step": s,
                "n": n,
                "self_matches_truth_pct": C.pct((sub.pred == sub.truth).mean()) if n else None,
                "self_matches_consensus_pct": (
                    C.pct((sub.pred == sub.consensus).mean()) if n else None
                ),
            }
        )
    return rows


RES["self_matches_consensus_by_step"] = {t: consensus_by_step(t) for t in MODELS}


# ============================================================
# ERROR PROPAGATION ALONG THE TRAJECTORY
# ============================================================


def _propagation(m):
    df = pd.DataFrame(
        [(mz, s, c) for (mz, s), c in C.SELF[m].items()], columns=["maze", "step", "correct"]
    )
    nxt = df.assign(step=df.step - 1)  # align k+1 onto k
    pairs = df.merge(nxt, on=["maze", "step"], suffixes=("", "_next"))
    pairs = pairs[pairs.step.between(1, 7)]
    given_c = pairs[pairs.correct]["correct_next"]
    given_w = pairs[~pairs.correct]["correct_next"]
    return {
        "p_correct_next_given_correct": C.pct(given_c.mean()) if len(given_c) else None,
        "p_correct_next_given_wrong": C.pct(given_w.mean()) if len(given_w) else None,
        "n_consecutive_pairs": int(len(pairs)),
    }


RES["error_propagation"] = {m: _propagation(m) for m in MODELS}


# ============================================================
# PREDICTABILITY HORIZON (first step below 50%)
# ============================================================

_pt = C.RECORDS[C.RECORDS.kind.isin(("self", "cross"))]
_step_acc = _pt.groupby(["target", "predictor", "step"], sort=False)["correct"].mean().mul(100.0)

horizon = {}
for t in MODELS:
    per_pred = _step_acc.xs(t, level="target").groupby("step").mean().reindex(range(1, 9))
    line = [float(v) if pd.notna(v) else None for v in per_pred]
    first_below = next((s for s, v in zip(range(1, 9), line) if v is not None and v < 50), None)
    horizon[t] = {
        "first_step_below_50pct": first_below,
        "predictability_by_step": [round(v) if v is not None else None for v in line],
    }
RES["predictability_horizon"] = horizon


# ============================================================
# WRITE + SUMMARY
# ============================================================

with open(os.path.join(OUT, "per_step.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print("self-consistency by step (fraction of validation cells whose runs agree):")
    for m in MODELS:
        print(
            f"  {m:7} "
            + " ".join(
                f"s{s}:{RES['self_consistency_by_step'][m][s]['agree_frac']}" for s in range(1, 9)
            )
        )
    print("\npredictability horizon (first step below 50% mean predictability):")
    for t, d in horizon.items():
        print(
            f"  {t:7} first<50% at step {d['first_step_below_50pct']}  curve={d['predictability_by_step']}"
        )
    print("\nerror propagation: P(next correct | this correct) vs P(next correct | this wrong):")
    for m, d in RES["error_propagation"].items():
        print(
            f"  {m:7} given-correct {d['p_correct_next_given_correct']}%  given-wrong {d['p_correct_next_given_wrong']}%  (n={d['n_consecutive_pairs']})"
        )
    print("\nself matches consensus vs truth, by step (opus):")
    for r in RES["self_matches_consensus_by_step"]["opus"]:
        print(
            f"  step {r['step']}: truth {r['self_matches_truth_pct']}%  consensus {r['self_matches_consensus_pct']}%  (n={r['n']})"
        )
    print("\n-> wrote results/per_step.json")
