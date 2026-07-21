#!/usr/bin/env python3
"""
Statistical rigor for the headline claims
=========================================

Covers the paired tests, bootstrap CIs, noise floor, max-selection-bias fix, baseline
sensitivity, and the conditional self-advantage at branch steps -- the inferential backbone
for the self-vs-other and Opus mid-horizon claims.

McNemar uses statsmodels' exact test; gap CIs use scipy's paired bootstrap.
Output: analysis/results/stats.json
"""

import json
import os
import collections
import statistics as st

import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
_RNG = np.random.default_rng(20260609)  # reproducible bootstrap
B = 2000  # bootstrap resamples
MODELS = C.MODELS
RES = {"metadata": C.metadata("stats")}


# ============================================================
# PAIRED HELPERS
# ============================================================


def paired(sa, sb, restrict=None):
    """Aligned (self_bool, other_bool) over shared (maze,step) keys, optionally filtered."""
    keys = set(sa) & set(sb)
    if restrict is not None:
        keys = {k for k in keys if restrict(k)}
    return [(sa[k], sb[k]) for k in sorted(keys)]


def mcnemar(pairs):
    """Exact two-sided McNemar on discordant pairs (statsmodels). Returns b, c, n, p."""
    b = sum(1 for x, y in pairs if x and not y)  # self right, other wrong
    c = sum(1 for x, y in pairs if y and not x)  # self wrong, other right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p = _mcnemar([[0, b], [c, 0]], exact=True).pvalue
    return {"b": b, "c": c, "n_discordant": n, "p_value": round(float(min(1.0, p)), 4)}


def boot_gap_ci(pairs):
    """Paired bootstrap 95% CI for (self_acc - other_acc) in percentage points."""
    if not pairs:
        return {"gap": None, "ci_lo": None, "ci_hi": None, "n": 0}
    n = len(pairs)
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    point = float(100.0 * (x.mean() - y.mean()))
    if n < 2 or ((x == x[0]).all() and (y == y[0]).all()):
        return {"gap": round(point, 1), "ci_lo": round(point, 1), "ci_hi": round(point, 1), "n": n}
    res = bootstrap(
        (x, y),
        lambda a, b: 100.0 * (a.mean() - b.mean()),
        paired=True,
        vectorized=False,
        n_resamples=B,
        confidence_level=0.95,
        method="percentile",
        rng=_RNG,
    )
    return {
        "gap": round(point, 1),
        "ci_lo": round(float(res.confidence_interval.low), 1),
        "ci_hi": round(float(res.confidence_interval.high), 1),
        "n": n,
    }


def best_other_model(t):
    """The single external model with the highest native cross-accuracy on target t."""
    cand = {p: C.acc(C.CROSS[(p, t)])[0] for p in MODELS if p != t and (p, t) in C.CROSS}
    cand = {p: v for p, v in cand.items() if v is not None}
    return max(cand, key=cand.get) if cand else None


# ============================================================
# SELF VS BEST-OTHER: PAIRED TEST + CI
# ============================================================

sib = {}
for t in MODELS:
    bo = best_other_model(t)
    if bo is None:
        continue
    pairs = paired(C.SELF[t], C.CROSS[(bo, t)])
    branch_pairs = paired(
        C.SELF[t], C.CROSS[(bo, t)], restrict=lambda k: C.is_branch(t, k[0], k[1])
    )
    sib[t] = {
        "best_other": bo,
        "overall": {**boot_gap_ci(pairs), **mcnemar(pairs)},
        "branch_only": {**boot_gap_ci(branch_pairs), **mcnemar(branch_pairs)},
    }
RES["self_vs_best_other_paired"] = sib


# ============================================================
# PER-STEP SELF VS BEST-OTHER (CI per step)
# ============================================================

perstep = {}
for t in MODELS:
    bo = best_other_model(t)
    if bo is None:
        continue
    rows = []
    for s in range(1, 9):
        pairs = paired(C.SELF[t], C.CROSS[(bo, t)], restrict=lambda k, ss=s: k[1] == ss)
        ci = boot_gap_ci(pairs)
        rows.append(
            {
                "step": s,
                **ci,
                "sig": bool(ci["ci_lo"] is not None and (ci["ci_lo"] > 0 or ci["ci_hi"] < 0)),
            }
        )
    perstep[t] = {"best_other": bo, "by_step": rows}
RES["self_vs_best_other_per_step"] = perstep


# ============================================================
# MAX-SELECTION-BIAS FIX: SELF'S RANK
# ============================================================

rank = {}
for t in MODELS:
    scores = {"self": C.acc(C.SELF[t])[0]}
    for p in MODELS:
        if p != t and (p, t) in C.CROSS:
            v = C.acc(C.CROSS[(p, t)])[0]
            if v is not None:
                scores[p] = v
    order = sorted(scores, key=scores.get, reverse=True)  # best first
    others = [v for k, v in scores.items() if k != "self"]
    rank[t] = {
        "self_acc": round(scores["self"], 1),
        "self_rank": order.index("self") + 1,  # 1 = best predictor of t
        "n_predictors": len(scores),
        "mean_other": round(st.mean(others), 1),
        "gap_vs_mean": round(scores["self"] - st.mean(others), 1),
    }
RES["self_rank_among_predictors"] = rank


# ============================================================
# VALIDATION-RUN NOISE FLOOR (run 1,2)
# ============================================================

noise = {}
for m in MODELS:
    cells = {}
    for mz, s, ri, rec in C.self_records(m, "reasoning"):
        if rec.get("parsed_position") is None:
            continue
        cells.setdefault((mz, s), {})[ri] = tuple(rec["parsed_position"])
    multi = {k: v for k, v in cells.items() if len(v) >= 2}  # the validation subsample
    by_step = {}
    agree_total = 0
    for (mz, s), runs in multi.items():
        ok = len(set(runs.values())) == 1
        agree_total += ok
        d = by_step.setdefault(s, [0, 0])
        d[0] += ok
        d[1] += 1
    noise[m] = {
        "n_validation_cells": len(multi),
        "frac_all_runs_agree": round(agree_total / len(multi), 3) if multi else None,
        "per_step_agreement": {s: round(v[0] / v[1], 3) for s, v in sorted(by_step.items())},
    }
RES["validation_self_consistency"] = noise


# ============================================================
# BASELINE SENSITIVITY
# ============================================================


# For each target/step on the target's consistent set, the chance baseline under three
# definitions, and self-accuracy's lift over each.
def modal_baseline(t, s, mazeset):
    """Accuracy of always predicting the single most common actual step-s position."""
    actual = [tuple(C.TRUTH[t][mz][s]) for mz in mazeset if s < len(C.TRUTH[t][mz])]
    if not actual:
        return None
    cnt = collections.Counter(actual).most_common(1)[0][1]
    return 100.0 * cnt / len(actual)


def uniform_baseline(t, s, mazeset, parity):
    """Mean over mazes of 1/|reachable cells at step s| (* 100)."""
    vals = []
    for mz in mazeset:
        if s >= len(C.TRUTH[t][mz]):
            continue
        reach = C.reachable_exactly(mz, s) if parity else C.reachable_shortest(mz, s)
        if reach:
            vals.append(100.0 / len(reach))
    return st.mean(vals) if vals else None


baseline = {}
for t in MODELS:
    mset = C.CONSISTENT[t]
    rows = []
    for s in range(1, 9):
        self_a = C.acc(C.SELF[t], None, s)[0]
        u_walk = uniform_baseline(t, s, mset, parity=True)
        u_short = uniform_baseline(t, s, mset, parity=False)
        modal = modal_baseline(t, s, mset)
        rows.append(
            {
                "step": s,
                "self": round(self_a) if self_a is not None else None,
                "uniform_walk": round(u_walk, 1) if u_walk is not None else None,
                "uniform_shortest": round(u_short, 1) if u_short is not None else None,
                "modal_nav": round(modal, 1) if modal is not None else None,
                "lift_vs_uniform_walk": (
                    round(self_a - u_walk, 1)
                    if (self_a is not None and u_walk is not None)
                    else None
                ),
                "lift_vs_modal": (
                    round(self_a - modal, 1) if (self_a is not None and modal is not None) else None
                ),
            }
        )
    baseline[t] = rows
RES["baseline_sensitivity"] = baseline


# ============================================================
# WHERE THE SELF-ADVANTAGE LIVES
# ============================================================

# Split branch decisions by whether the TARGET took the prior (alphabetically-first) direction.
# Genuine (if narrow) self-knowledge should surface on idiosyncratic branches, where the target
# departs from the generic prior and an outsider has nothing but the prior to go on.
prior_split = {}
for t in MODELS:
    bo = best_other_model(t)
    if bo is None:
        continue
    keys = sorted(set(C.SELF[t]) & set(C.CROSS[(bo, t)]))
    buckets = {"prior_aligned": [], "idiosyncratic": []}
    for k in keys:
        if not C.is_branch(t, k[0], k[1]):
            continue
        fl = C.chose_first_listed(t, k[0], k[1])
        if fl is None:
            continue
        buckets["prior_aligned" if fl else "idiosyncratic"].append(
            (C.SELF[t][k], C.CROSS[(bo, t)][k])
        )
    entry = {"best_other": bo}
    for name, pairs in buckets.items():
        entry[name] = {
            "self_acc": C.pct(sum(x for x, _ in pairs), len(pairs)),
            "other_acc": C.pct(sum(y for _, y in pairs), len(pairs)),
            **boot_gap_ci(pairs),
            **mcnemar(pairs),
        }
    prior_split[t] = entry
RES["self_advantage_prior_aligned_vs_idiosyncratic"] = prior_split


# ============================================================
# IS RUN-STABILITY A WITHIN-MODEL CORRECTNESS SIGNAL?
# ============================================================

# On the validation subsample, are predictions that are stable across runs more often correct?
consistency_acc = {}
for m in MODELS:
    cells = {}
    for mz, s, ri, rec in C.self_records(m, "reasoning"):
        if rec.get("parsed_position") is None:
            continue
        cells.setdefault((mz, s), {})[ri] = tuple(rec["parsed_position"])
    a_ok = a_n = u_ok = u_n = 0
    for k, runs in cells.items():
        if len(runs) < 2 or k not in C.SELF[m]:  # validation cells with a scored run-0
            continue
        if len(set(runs.values())) == 1:
            a_n += 1
            a_ok += C.SELF[m][k]
        else:
            u_n += 1
            u_ok += C.SELF[m][k]
    consistency_acc[m] = {
        "stable_acc": C.pct(a_ok, a_n) if a_n else None,
        "n_stable": a_n,
        "unstable_acc": C.pct(u_ok, u_n) if u_n else None,
        "n_unstable": u_n,
    }
RES["self_consistency_vs_accuracy"] = consistency_acc


# ============================================================
# SELF-ACCURACY BY MOVE TYPE, REASONING VS NO-REASONING
# ============================================================


# The A5 split: each model's self-accuracy on atypical / default / determined cells, with and
# without a reasoning trace. "Determined" = non-branch cells (one unvisited move).
def _move_type(t, mz, step):
    if not C.is_branch(t, mz, step):
        return "determined"
    return "default" if C.chose_first_listed(t, mz, step) else "atypical"


by_move_type = {}
for t in MODELS:
    out = {}
    for kind, scored in (("reason", C.SELF[t]), ("nr", C.SELF_NR[t])):
        buckets = {"atypical": [], "default": [], "determined": []}
        for (mz, step), okv in scored.items():
            buckets[_move_type(t, mz, step)].append(okv)
        for k, v in buckets.items():
            out.setdefault(k, {})[kind] = C.pct(sum(v), len(v))
            out[k]["n_" + kind] = len(v)
    by_move_type[t] = out
RES["self_by_move_type_reasoning_vs_nr"] = by_move_type


# ============================================================
# ATYPICAL-CELL SENSITIVITY: PER-PREDICTOR AND ROTATING-BEST COMPARATORS
# ============================================================


# Robustness of the atypical-cell self-advantage to the choice of comparator. Per-predictor:
# self vs each cross-predictor individually. Rotating-best: at each step, the opponent is the
# cross-predictor with the highest accuracy on ALL of the target's cells at that step (a harsher,
# selection-biased comparator), evaluated on the atypical cells only.
def _atypical_cells(t):
    return sorted(
        (mz, step)
        for (mz, step) in C.SELF[t]
        if C.is_branch(t, mz, step) and not C.chose_first_listed(t, mz, step)
    )


def _step_best_opponent(t, step):
    best, bacc = None, -1.0
    for p in MODELS:
        if p == t:
            continue
        vals = [v for (mz, st2), v in C.CROSS[(p, t)].items() if st2 == step]
        if vals and 100.0 * sum(vals) / len(vals) > bacc:
            bacc, best = 100.0 * sum(vals) / len(vals), p
    return best


atyp_sensitivity = {}
for t in MODELS:
    cells = _atypical_cells(t)
    per_pred = {}
    for p in MODELS:
        if p == t:
            continue
        pairs = [(C.SELF[t][k], C.CROSS[(p, t)][k]) for k in cells if k in C.CROSS[(p, t)]]
        mc = mcnemar(pairs)
        per_pred[p] = {
            "gap": C.pct(sum(x for x, _ in pairs), len(pairs))
            - C.pct(sum(y for _, y in pairs), len(pairs)),
            "gap_exact": round(
                100.0 * (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / len(pairs), 1
            ),
            "p_value": mc["p_value"],
            "n": len(pairs),
        }
    rot_pairs = []
    opp_by_step = {}
    for step in sorted(set(k[1] for k in cells)):
        opp = _step_best_opponent(t, step)
        opp_by_step[step] = opp
        rot_pairs += [
            (C.SELF[t][k], C.CROSS[(opp, t)][k])
            for k in cells
            if k[1] == step and k in C.CROSS[(opp, t)]
        ]
    mc = mcnemar(rot_pairs)
    atyp_sensitivity[t] = {
        "per_predictor": per_pred,
        "rotating_best": {
            "gap": round(
                100.0
                * (sum(x for x, _ in rot_pairs) - sum(y for _, y in rot_pairs))
                / len(rot_pairs),
                1,
            ),
            "p_value": mc["p_value"],
            "n": len(rot_pairs),
            "opponent_by_step": opp_by_step,
        },
    }
RES["atypical_comparator_sensitivity"] = atyp_sensitivity


# ============================================================
# PER-STEP McNEMAR WITH HOLM CORRECTION (FIXED AND ROTATING OPPONENT)
# ============================================================


# Two versions of "is any single step significant after multiple testing": against the fixed
# best-overall opponent, and against the per-step rotating best opponent (harsher).
def _holm(raw):
    order = sorted(raw, key=lambda k: raw[k])
    out, running = {}, 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[k]))
        out[k] = round(running, 4)
    return out


per_step_holm = {}
for t in MODELS:
    bo = best_other_model(t)
    fixed_raw = {}
    rot_raw = {}
    for step in range(1, 9):
        ks = sorted(k for k in set(C.SELF[t]) & set(C.CROSS[(bo, t)]) if k[1] == step)
        fixed_raw[step] = mcnemar([(C.SELF[t][k], C.CROSS[(bo, t)][k]) for k in ks])["p_value"]
        opp = _step_best_opponent(t, step)
        ks2 = sorted(k for k in set(C.SELF[t]) & set(C.CROSS[(opp, t)]) if k[1] == step)
        rot_raw[step] = mcnemar([(C.SELF[t][k], C.CROSS[(opp, t)][k]) for k in ks2])["p_value"]
    per_step_holm[t] = {
        "fixed_opponent": {"opponent": bo, "raw": fixed_raw, "holm": _holm(fixed_raw)},
        "rotating_opponent": {"raw": rot_raw, "holm": _holm(rot_raw)},
    }
RES["per_step_mcnemar_holm"] = per_step_holm


# ============================================================
# NO-REASONING UNIQUE INFORMATION
# ============================================================

# Cells where the no-reasoning self-prediction is correct while all four (reasoning)
# cross-predictors are wrong. No-reasoning cross-predictions were not collected, so the
# reasoning cross-predictors are the only available comparator.
nr_unique = {}
for t in MODELS:
    n = only = 0
    for (mz, step), okv in C.SELF_NR[t].items():
        others = [C.CROSS[(p, t)].get((mz, step)) for p in MODELS if p != t]
        others = [x for x in others if x is not None]
        if len(others) < 4:
            continue
        n += 1
        if okv and not any(others):
            only += 1
    nr_unique[t] = {"only": only, "n": n, "pct": C.pct(only, n)}
RES["nr_unique_info"] = nr_unique


# ============================================================
# DETERMINED-CELL SELF-ADVANTAGE
# ============================================================

# Completes the move-type split: the paired self-vs-best-other gap on determined (non-branch)
# cells, same methodology as the prior-aligned / idiosyncratic split above.
det = {}
for t in MODELS:
    bo = best_other_model(t)
    pairs = paired(
        C.SELF[t], C.CROSS[(bo, t)], restrict=lambda k, tt=t: not C.is_branch(tt, k[0], k[1])
    )
    ci = boot_gap_ci(pairs)
    det[t] = {**ci, **mcnemar(pairs), "best_other": bo}
RES["self_advantage_determined"] = det


# ============================================================
# WRITE + SUMMARY
# ============================================================

with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print("self vs best-other (paired, native): gap [95% CI], McNemar p")
    for t, d in sib.items():
        o = d["overall"]
        print(
            f"  {t:7} vs {d['best_other']:6}: {o['gap']:+5.1f} [{o['ci_lo']:+.1f}, {o['ci_hi']:+.1f}]  p={o['p_value']}  (n={o['n']})"
        )
    print("\nself rank among all predictors (1 = best predictor of that target):")
    for t, d in rank.items():
        print(
            f"  {t:7} rank {d['self_rank']}/{d['n_predictors']}  (self {d['self_acc']}, mean-other {d['mean_other']}, gap_vs_mean {d['gap_vs_mean']:+.1f})"
        )
    print("\nOpus per-step self vs best-other (CI), looking for steps excluding 0:")
    for r in perstep["opus"]["by_step"]:
        flag = "  <-- sig" if r["sig"] else ""
        print(f"  step {r['step']}: {r['gap']:+5.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]{flag}")
    print("\nvalidation self-consistency (temp-0 noise floor):")
    for m, d in noise.items():
        print(
            f"  {m:7} {d['frac_all_runs_agree']} of {d['n_validation_cells']} validation cells fully agree"
        )
    print("\nwhere the self-advantage lives - self vs best-other at branches, by prior-alignment:")
    for t, d in prior_split.items():
        pa, idi = d["prior_aligned"], d["idiosyncratic"]
        print(
            f"  {t:7} prior-aligned gap={pa['gap']:+.1f} (self {pa['self_acc']} vs {pa['other_acc']}, n={pa['n']}, p={pa['p_value']})"
            f"  | idiosyncratic gap={idi['gap']:+.1f} (self {idi['self_acc']} vs {idi['other_acc']}, n={idi['n']}, p={idi['p_value']})"
        )
    print("\nrun-to-run stability vs correctness (validation subsample):")
    for m, d in consistency_acc.items():
        print(
            f"  {m:7} stable: {d['stable_acc']}% (n={d['n_stable']})  unstable: {str(d['unstable_acc']) + '%' if d['unstable_acc'] is not None else 'n/a'} (n={d['n_unstable']})"
        )
    print("\n-> wrote results/stats.json")
