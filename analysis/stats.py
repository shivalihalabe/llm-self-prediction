#!/usr/bin/env python3
"""
Statistical rigor for the headline claims
=========================================

Covers the paired tests, bootstrap CIs, noise floor, max-selection-bias fix, baseline
sensitivity, and the conditional self-advantage at branch steps -- the inferential backbone
for the self-vs-other and Opus mid-horizon claims.

Cells within a maze are one reconstruction, not independent trials, so every paired test
clusters at the maze level: gap CIs come from a cluster bootstrap over mazes, and paired
p-values from a maze-level sign-flip permutation of the signed discordance.

Output: analysis/results/stats.json
"""

import collections
import json
import os
import statistics as st

import numpy as np

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
B = 2000  # bootstrap resamples
N_PERM = 10000  # sign-flip / label-shuffle permutation draws
SEED = 20260609  # a fresh generator is constructed per call, so test order cannot matter
MODELS = C.MODELS
RES = {"metadata": C.metadata("stats")}


# ============================================================
# PAIRED HELPERS
# ============================================================


def paired(sa, sb, restrict=None):
    """Aligned (maze, self_bool, other_bool) over shared (maze, step) keys.

    The maze id is the cluster label for the bootstrap and permutation below.
    """
    keys = set(sa) & set(sb)
    if restrict is not None:
        keys = {k for k in keys if restrict(k)}
    return [(k[0], sa[k], sb[k]) for k in sorted(keys)]


def _by_maze(pairs):
    """{maze: [(self_bool, other_bool), ...]} in sorted maze order."""
    out = collections.OrderedDict()
    for mz, x, y in pairs:
        out.setdefault(mz, []).append((x, y))
    return out


def _sign_flip_p(pairs):
    """Raw two-sided p from a maze-level sign-flip permutation of the signed discordance.

    Each maze contributes sum(self_correct - other_correct) over its cells; N_PERM random
    sign vectors over mazes give the null distribution of the absolute total.
    """
    if not pairs:
        return 1.0
    d = np.array([sum(x - y for x, y in cells) for cells in _by_maze(pairs).values()], dtype=float)
    observed = abs(d.sum())
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1, 1], size=(N_PERM, len(d)))
    return float((np.abs(signs @ d) >= observed).mean())


def mcnemar(pairs):
    """Discordant counts plus the maze-level sign-flip permutation p-value."""
    b = sum(1 for _, x, y in pairs if x and not y)  # self right, other wrong
    c = sum(1 for _, x, y in pairs if y and not x)  # self wrong, other right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value_cluster_perm": 1.0}
    return {"b": b, "c": c, "n_discordant": n, "p_value_cluster_perm": C.fmt_p(_sign_flip_p(pairs))}


def boot_gap_ci(pairs):
    """Maze-level cluster bootstrap 95% CI for (self_acc - other_acc) in percentage points.

    Resamples the maze list with replacement and takes all cells of each drawn maze.
    """
    if not pairs:
        return {"gap": None, "ci_lo": None, "ci_hi": None, "n": 0}
    n = len(pairs)
    x = np.array([p[1] for p in pairs], dtype=float)
    y = np.array([p[2] for p in pairs], dtype=float)
    point = float(100.0 * (x.mean() - y.mean()))
    if n < 2 or ((x == x[0]).all() and (y == y[0]).all()):
        return {"gap": round(point, 1), "ci_lo": round(point, 1), "ci_hi": round(point, 1), "n": n}
    groups = _by_maze(pairs)
    sx = np.array([sum(x2 for x2, _ in cells) for cells in groups.values()], dtype=float)
    sy = np.array([sum(y2 for _, y2 in cells) for cells in groups.values()], dtype=float)
    sn = np.array([len(cells) for cells in groups.values()], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(sx), size=(B, len(sx)))
    tn = sn[draws].sum(axis=1)
    gaps = 100.0 * (sx[draws].sum(axis=1) - sy[draws].sum(axis=1)) / tn
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "gap": round(point, 1),
        "ci_lo": round(float(lo), 1),
        "ci_hi": round(float(hi), 1),
        "n": n,
    }


best_other_model = C.best_other_model  # defined once in common.py


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
        C.SELF[t], C.CROSS[(bo, t)], restrict=lambda k, tt=t: C.is_branch(tt, k[0], k[1])
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
# Every per-step cell has exactly one observation per maze, so the maze-level cluster
# bootstrap is the same method as the plain paired bootstrap here; the shared code path is
# used anyway so there is one set of test functions, not two.


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
                "ci_excludes_zero": bool(
                    ci["ci_lo"] is not None and (ci["ci_lo"] > 0 or ci["ci_hi"] < 0)
                ),
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
    return "default" if C.chose_first_unvisited(t, mz, step) else "atypical"


by_move_type = {}
for t in MODELS:
    out = {}
    for kind, scored in (("reason", C.SELF[t]), ("nr", C.SELF_NR[t])):
        buckets = {"atypical": [], "default": [], "determined": []}
        for (mz, step), okv in scored.items():
            buckets[_move_type(t, mz, step)].append(okv)
        for k, v in buckets.items():
            out.setdefault(k, {})[kind] = C.pct(sum(v), len(v))
            out[k]["n_correct_" + kind] = sum(v)
            out[k]["n_" + kind] = len(v)
    by_move_type[t] = out
RES["self_by_move_type_reasoning_vs_nr"] = by_move_type


# ============================================================
# PER-STEP McNEMAR WITH HOLM CORRECTION (FIXED AND ROTATING OPPONENT)
# ============================================================
# Two versions of "is any single step significant after multiple testing": against the fixed
# best-overall opponent, and against the per-step rotating best opponent (harsher). One cell
# per maze at each step, so the maze-level sign-flip is the plain sign-flip here.


def _holm(raw):
    order = sorted(raw, key=lambda k: raw[k])
    out, running = {}, 0.0
    for rank_i, k in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank_i) * raw[k]))
        out[k] = C.fmt_p(running)
    return out


def _step_best_opponent(t, step):
    best, bacc = None, -1.0
    for p in MODELS:
        if p == t:
            continue
        vals = [v for (mz, st2), v in C.CROSS[(p, t)].items() if st2 == step]
        if vals and 100.0 * sum(vals) / len(vals) > bacc:
            bacc, best = 100.0 * sum(vals) / len(vals), p
    return best


per_step_holm = {}
for t in MODELS:
    bo = best_other_model(t)
    fixed_raw = {}
    rot_raw = {}
    for step in range(1, 9):
        fixed_raw[step] = _sign_flip_p(
            paired(C.SELF[t], C.CROSS[(bo, t)], restrict=lambda k, ss=step: k[1] == ss)
        )
        opp = _step_best_opponent(t, step)
        rot_raw[step] = _sign_flip_p(
            paired(C.SELF[t], C.CROSS[(opp, t)], restrict=lambda k, ss=step: k[1] == ss)
        )
    per_step_holm[t] = {
        "fixed_opponent": {
            "opponent": bo,
            "raw": {s: C.fmt_p(v) for s, v in fixed_raw.items()},
            "holm": _holm(fixed_raw),
        },
        "rotating_opponent": {
            "raw": {s: C.fmt_p(v) for s, v in rot_raw.items()},
            "holm": _holm(rot_raw),
        },
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
# ATYPICAL AND DEFAULT SELF-ADVANTAGE (PER-PREDICTOR, BEST, MEAN, HOLM)
# ============================================================
# A decision point has two or more unvisited legal moves; default means the target took the
# alphabetically-first unvisited direction, atypical anything else. For each target and each
# move type: self accuracy, the full per-predictor vector, the best single comparator and the
# mean comparator, each with unambiguous field names. Holm-adjusted p-values are given across
# the five per-target best-comparator tests. Correct/total counts accompany every accuracy so
# the gaps (computed unrounded) are checkable against the emitted figures.


def _split_cells(t):
    out = {"atypical": [], "default": []}
    for mz, step in sorted(C.SELF[t]):
        if not C.is_branch(t, mz, step):
            continue
        out["default" if C.chose_first_unvisited(t, mz, step) else "atypical"].append((mz, step))
    return out


def _advantage(t, cells):
    self_vals = [C.SELF[t][k] for k in cells]
    n_self_correct = sum(self_vals)
    self_raw = 100.0 * n_self_correct / len(self_vals) if cells else 0.0
    per_pred = {}
    raw = {}
    raw_p = {}
    for p in MODELS:
        if p == t:
            continue
        pairs = [(k[0], C.SELF[t][k], C.CROSS[(p, t)][k]) for k in cells if k in C.CROSS[(p, t)]]
        # self accuracy over the same paired subset as the comparator; identical to self_raw
        # today because every cross predictor answers every self cell, but this prevents a
        # silent unpaired gap if that ever stops holding
        self_sub = 100.0 * sum(x for _, x, _ in pairs) / len(pairs) if pairs else 0.0
        raw[p] = 100.0 * sum(y for _, _, y in pairs) / len(pairs) if pairs else 0.0
        raw_p[p] = _sign_flip_p(pairs)
        mc = mcnemar(pairs)
        per_pred[p] = {
            "acc": round(raw[p], 1),
            "n_correct": sum(y for _, _, y in pairs),
            "gap": round(self_sub - raw[p], 1),
            "p_value_cluster_perm": mc["p_value_cluster_perm"],
            "n_discordant": mc["n_discordant"],
            "n": len(pairs),
        }
    best = max(raw, key=raw.get)
    mean_raw = sum(raw.values()) / len(raw)
    return {
        "n": len(cells),
        "self_acc": round(self_raw, 1),
        "self_n_correct": n_self_correct,
        "per_predictor": per_pred,
        "best_other": {
            "model": best,
            "acc": per_pred[best]["acc"],
            "gap_vs_best_other": per_pred[best]["gap"],
            "p_value_cluster_perm": per_pred[best]["p_value_cluster_perm"],
            "n_discordant": per_pred[best]["n_discordant"],
        },
        "mean_other": {
            "acc": round(mean_raw, 1),
            "gap_vs_mean_other": round(self_raw - mean_raw, 1),
        },
        "_raw_best_p": raw_p[best],
    }


adv = {"atypical": {}, "default": {}}
for t in MODELS:
    cells = _split_cells(t)
    for kind in ("atypical", "default"):
        adv[kind][t] = _advantage(t, cells[kind])
for kind in ("atypical", "default"):
    raw = {t: adv[kind][t].pop("_raw_best_p") for t in MODELS}
    adv[kind]["holm_adjusted_best_other"] = _holm(raw)
RES["self_advantage_by_move_type"] = adv


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
# FIRST VS LATER ATYPICAL CELLS (ACCUMULATED-ERROR CHECK)
# ============================================================
# A prediction's accuracy at step N depends on the whole reconstruction from step 1, so a cell
# labeled atypical can be wrong because of an error several steps earlier. This splits each
# target's atypical cells by whether the step is the first atypical move in its maze or a later
# one (where the reconstruction had already had an opportunity to diverge). If the pooled
# self-advantage were an artifact of accumulated error it would appear in the later group; it
# appears in the first group only. The first bucket has one cell per maze by construction; it runs
# through the same clustered code path for consistency.


first_later = {}
for t in MODELS:
    per_maze = {}
    for mz, step in sorted(C.SELF[t]):
        if C.is_branch(t, mz, step) and not C.chose_first_unvisited(t, mz, step):
            per_maze.setdefault(mz, []).append(step)
    first, later = [], []
    for mz, steps in per_maze.items():
        steps = sorted(steps)
        first.append((mz, steps[0]))
        later += [(mz, step) for step in steps[1:]]
    first_later[t] = {
        "first": _advantage(t, sorted(first)) if first else {"n": 0},
        "later": _advantage(t, sorted(later)) if later else {"n": 0},
    }
    for part in first_later[t].values():
        part.pop("_raw_best_p", None)
RES["atypical_first_vs_later"] = first_later


# ============================================================
# ONE-UNVISITED CELLS
# ============================================================
# One-unvisited cells (2+ legal moves but a single unvisited one) are excluded from decision
# points: prediction there is far from ceiling, but no model shows a positive significant
# self-advantage, so the cells carry no model-specific signature.


def _one_unvisited_cells(t):
    out = []
    for mz, step in sorted(C.SELF[t]):
        legal = C.legal_moves(t, mz, step)
        if len(legal) >= 2 and len(C.unvisited_moves(t, mz, step)) == 1:
            out.append((mz, step))
    return out


one_unv = {}
for t in MODELS:
    one_unv[t] = _advantage(t, _one_unvisited_cells(t))
    one_unv[t].pop("_raw_best_p", None)
RES["one_unvisited_self_advantage"] = one_unv


# ============================================================
# UNIQUE INFORMATION (A3)
# ============================================================
# On the cells every predictor of a target answered, the cells where exactly one of the five
# predictors (self plus four cross) is correct and the other four are wrong. Self is compared
# against each cross-predictor with the maze-level sign-flip permutation; "self beats every
# predictor" is an intersection-union claim, so the maximum p is valid without multiplicity
# adjustment across the four comparators. The bootstrap CI is on the unique-correct count
# difference against the strongest cross-predictor, clustered by maze.


def _boot_count_diff_ci(pairs):
    """Cluster-bootstrap 95% CI on sum(x) - sum(y) over mazes (counts, not percentages)."""
    groups = _by_maze(pairs)
    sx = np.array([sum(x2 for x2, _ in cells) for cells in groups.values()], dtype=float)
    sy = np.array([sum(y2 for _, y2 in cells) for cells in groups.values()], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(sx), size=(B, len(sx)))
    diffs = sx[draws].sum(axis=1) - sy[draws].sum(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 1), round(float(hi), 1)


unique_info = {}
for t in MODELS:
    preds = {"self": C.SELF[t]}
    for p in MODELS:
        if p != t and (p, t) in C.CROSS:
            preds[p] = C.CROSS[(p, t)]
    keys = sorted(set.intersection(*(set(d) for d in preds.values())))
    unique = {
        nm: {
            k: preds[nm][k] and not any(preds[o][k] for o in preds if o != nm) for k in keys
        }
        for nm in preds
    }
    counts = {nm: sum(u.values()) for nm, u in unique.items()}
    p_vs = {}
    raw_ps = {}
    for p in preds:
        if p == "self":
            continue
        pairs = [(k[0], unique["self"][k], unique[p][k]) for k in keys]
        raw_ps[p] = _sign_flip_p(pairs)
        p_vs[p] = C.fmt_p(raw_ps[p])
    cross_counts = {p: c for p, c in counts.items() if p != "self"}
    strongest = max(cross_counts, key=cross_counts.get)
    ci_lo, ci_hi = _boot_count_diff_ci(
        [(k[0], unique["self"][k], unique[strongest][k]) for k in keys]
    )
    unique_info[t] = {
        "n_cells": len(keys),
        "unique_correct": {(t if nm == "self" else nm): counts[nm] for nm in preds},
        "p_vs_each_cross": p_vs,
        "max_p": C.fmt_p(max(raw_ps.values())),
        "strongest_other": strongest,
        "count_diff_vs_strongest": counts["self"] - counts[strongest],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }
RES["unique_information"] = unique_info


# ============================================================
# ATYPICAL REASONING DEPENDENCE (A5)
# ============================================================
# On each target's atypical cells, the with-reasoning self-prediction paired against the
# no-reasoning one, clustered by maze. No-reasoning cross-predictions were never collected,
# so this measures the collapse of self-accuracy rather than of the advantage directly; the
# advantage claim follows by comparing the no-reasoning accuracy against what the best
# cross-predictor achieves on the same cells.


reasoning_dep = {}
for t in MODELS:
    cells = [k for k in _split_cells(t)["atypical"] if k in C.SELF_NR[t]]
    pairs = [(k[0], C.SELF[t][k], C.SELF_NR[t][k]) for k in cells]
    n = len(pairs)
    n_r = sum(x for _, x, _ in pairs)
    n_nr = sum(y for _, _, y in pairs)
    reasoning_dep[t] = {
        "n": n,
        "acc_reasoning": C.pct(n_r, n),
        "n_correct_reasoning": n_r,
        "acc_noreasoning": C.pct(n_nr, n),
        "n_correct_noreasoning": n_nr,
        "gap": round(100.0 * (n_r - n_nr) / n, 1) if n else None,
        "p_value_cluster_perm": C.fmt_p(_sign_flip_p(pairs)) if n else None,
    }
RES["atypical_reasoning_dependence"] = reasoning_dep


# ============================================================
# RUN-TO-RUN CONSISTENCY BY MOVE TYPE (D4)
# ============================================================
# On the validation subsample, does run-to-run instability concentrate on atypical cells?
# Label-shuffle permutation of the atypical/other split against the per-cell agreement flag.
# Every row here is underpowered (single-digit unstable-cell counts); the emitted counts are
# the point -- the validation subsample cannot resolve the question for any model.


consistency_mt = {}
for m in MODELS:
    cells = {}
    for mz, s, ri, rec in C.self_records(m, "reasoning"):
        if rec.get("parsed_position") is None:
            continue
        cells.setdefault((mz, s), {})[ri] = tuple(rec["parsed_position"])
    atyp_agree, other_agree = [], []
    for k, runs in cells.items():
        if len(runs) < 2 or k not in C.SELF[m]:
            continue
        agree = len(set(runs.values())) == 1
        if _move_type(m, k[0], k[1]) == "atypical":
            atyp_agree.append(agree)
        else:
            other_agree.append(agree)
    n_a, n_o = len(atyp_agree), len(other_agree)
    p = None
    if n_a and n_o:
        obs = abs(st.mean(atyp_agree) - st.mean(other_agree))
        flags = np.array(atyp_agree + other_agree, dtype=float)
        rng = np.random.default_rng(SEED)
        diffs = np.empty(N_PERM)
        for i in range(N_PERM):
            perm = rng.permutation(flags)
            diffs[i] = abs(perm[:n_a].mean() - perm[n_a:].mean())
        p = C.fmt_p(float((diffs >= obs - 1e-12).mean()))
    consistency_mt[m] = {
        "atypical_agree_pct": C.pct(sum(atyp_agree), n_a) if n_a else None,
        "n_atypical": n_a,
        "n_atypical_unstable": n_a - sum(atyp_agree),
        "other_agree_pct": C.pct(sum(other_agree), n_o) if n_o else None,
        "n_other": n_o,
        "n_other_unstable": n_o - sum(other_agree),
        "p_value_label_perm": p,
    }
RES["consistency_by_move_type"] = consistency_mt


# ============================================================
# WRITE
# ============================================================


with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(RES, f, indent=1)
