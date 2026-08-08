#!/usr/bin/env python3
"""
Statistical tests
---

Provides the paired tests, intervals and baselines behind the self-against-other
comparisons. Cells within a maze are one reconstruction rather than independent trials, so
every paired test clusters at the maze level.

Method:
- gap CIs from a cluster bootstrap that resamples mazes with replacement
- paired p-values from a maze-level sign-flip permutation of the signed discordance
- Holm adjustment across the five per-target tests

Measures:
- self against best-other, overall and per step
- self's rank among all five predictors
- the validation-run noise floor, and whether run-stability predicts correctness
- chance baselines under three definitions
- self-advantage by move type, and on determined and one-unvisited cells
- cross-predictor against cross-predictor on atypical cells, and the default-move ranking
- the atypical self-advantage recomputed from the alternate self-prediction runs
- unique information, reasoning dependence, and run-to-run consistency by move type
- whether departing from a predictor's own generic answer helps or hurts

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
SEED = 20260609  # a fresh generator is constructed per call, so test order can't matter
MODELS = C.MODELS
RES = {"metadata": C.metadata("stats")}


# Paired helpers

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


def sign_flip_test(pairs):
    """Discordance counts (b/c) for a paired comparison, with a maze-level sign-flip
    permutation p-value."""
    b = sum(1 for _, x, y in pairs if x and not y)  # self right, other wrong
    c = sum(1 for _, x, y in pairs if y and not x)  # self wrong, other right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value_cluster_perm": 1.0}
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value_cluster_perm": C.fmt_p(_sign_flip_p(pairs), N_PERM),
    }


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


# Self vs best-other: sign-flip permutation + cluster bootstrap CI

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
        "overall": {**boot_gap_ci(pairs), **sign_flip_test(pairs)},
        "branch_only": {**boot_gap_ci(branch_pairs), **sign_flip_test(branch_pairs)},
    }
RES["self_vs_best_other_paired"] = sib


# Per-step self vs best-other (CI per step)
# Every per-step cell has exactly one observation per maze, so the maze-level cluster
# bootstrap is the same method as the plain paired bootstrap here; the shared code path is
# used anyway so there's one set of test functions, not two.

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


# Max-selection-bias fix: self's rank

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


# Validation-run noise floor (run 1,2)

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


# Baseline sensitivity
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


# Run-stability as a correctness signal
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


# Self-accuracy by move type, reasoning vs no-reasoning
# The A5 split: each model's self-accuracy on atypical / default / determined cells, with and
# without a reasoning trace. "Determined" = non-branch cells (one unvisited move).

def _move_type(t, mz, step):
    """Classify a cell as determined, default, or atypical."""
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


# Per-step paired comparison with Holm correction (fixed and rotating opponent)
# One cell per maze here, so the maze-level sign flip is a sign test on the discordant pairs,
# which is McNemar's exact test evaluated by Monte Carlo rather than by the binomial.
# Two versions of "is any single step significant after multiple testing": against the fixed
# best-overall opponent, and against the per-step rotating best opponent (harsher). One cell
# per maze at each step, so the maze-level sign-flip is the plain sign-flip here.

def _holm(raw):
    """Holm step-down adjustment of a {key: p-value} mapping."""
    order = sorted(raw, key=lambda k: raw[k])
    out, running = {}, 0.0
    for rank_i, k in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank_i) * raw[k]))
        out[k] = C.fmt_p(running, N_PERM)
    return out


def _step_best_opponent(t, step):
    """The cross-predictor with the highest accuracy on a target at one step."""
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
            "raw": {s: C.fmt_p(v, N_PERM) for s, v in fixed_raw.items()},
            "holm": _holm(fixed_raw),
        },
        "rotating_opponent": {
            "raw": {s: C.fmt_p(v, N_PERM) for s, v in rot_raw.items()},
            "holm": _holm(rot_raw),
        },
    }
RES["per_step_paired_holm"] = per_step_holm


# No-reasoning unique information
# The five candidates in a cell are the target's own no-reasoning self-prediction and the four
# (reasoning) cross-predictions; a candidate scores when it is correct and the other four are
# wrong. No-reasoning cross-predictions were not collected, so the reasoning cross-predictors
# are the only available comparator. only and pct report the target's own count, unique_correct
# all five, keyed as in unique_information so the no-reasoning count reads against what each
# cross-predictor manages on the same cells. Cells with two correct answers or none have no
# unique winner, so the five counts don't sum to n.

nr_unique = {}
for t in MODELS:
    n = 0
    counts = {t: 0, **{p: 0 for p in MODELS if p != t}}
    for (mz, step), okv in C.SELF_NR[t].items():
        cand = {p: C.CROSS[(p, t)].get((mz, step)) for p in MODELS if p != t}
        cand = {p: x for p, x in cand.items() if x is not None}
        if len(cand) < 4:
            continue
        n += 1
        cand[t] = okv
        for nm, ok in cand.items():
            if ok and not any(v for o, v in cand.items() if o != nm):
                counts[nm] += 1
    nr_unique[t] = {
        "only": counts[t],
        "n": n,
        "pct": C.pct(counts[t], n),
        "unique_correct": counts,
    }
RES["nr_unique_info"] = nr_unique


# Atypical and default self-advantage (per-predictor, best, mean, Holm)
# A decision point has two or more unvisited legal moves; default means the target took the
# alphabetically-first unvisited direction, atypical anything else. For each target and each
# move type: self accuracy, the full per-predictor vector, the best single comparator and the
# mean comparator, each with unambiguous field names. Holm-adjusted p-values are given across
# the five per-target best-comparator tests. Correct/total counts accompany every accuracy so
# the gaps (computed unrounded) are checkable against the emitted figures.

def _split_cells(t):
    """A target's decision points, split into default and atypical."""
    out = {"atypical": [], "default": []}
    for mz, step in sorted(C.SELF[t]):
        if not C.is_branch(t, mz, step):
            continue
        out["default" if C.chose_first_unvisited(t, mz, step) else "atypical"].append((mz, step))
    return out


def _advantage(t, cells):
    """Self accuracy on a cell set against every cross-predictor, with clustered tests."""
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
        mc = sign_flip_test(pairs)
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


# Cross-predictor against cross-predictor on atypical cells
# The key above pits a target's own self-prediction against each cross-predictor. This pits
# the cross-predictors against each other on the identical cells, which the self-versus-other
# framing never reaches. Pairs run in MODELS order, so the sign of the gap is fixed: it is the
# first-named predictor's rate minus the second's.

atypical_pairs = {}
for t in MODELS:
    cells = _split_cells(t)["atypical"]
    others = [p for p in MODELS if p != t and (p, t) in C.CROSS]
    rows = {}
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            pr = [(mz, C.CROSS[(a, t)][(mz, s)], C.CROSS[(b, t)][(mz, s)]) for mz, s in cells]
            n_a = sum(x for _, x, _ in pr)
            n_b = sum(y for _, _, y in pr)
            flip = sign_flip_test(pr)
            rows[f"{a}|{b}"] = {
                "acc_a": C.pct(n_a, len(pr)),
                "acc_b": C.pct(n_b, len(pr)),
                **boot_gap_ci(pr),
                "n_discordant": flip["n_discordant"],
                "p_value_cluster_perm": flip["p_value_cluster_perm"],
            }
    atypical_pairs[t] = rows
RES["atypical_cross_vs_cross"] = atypical_pairs


# Default-move predictor ranking
# Derived from the default-cell accuracies above, with the target's own self-prediction ranked
# alongside the four cross-predictors. Rank 1 is the lowest accuracy, so the ranking reads as
# "who is worst at this target's rule-following moves".

default_rank = {}
for t in MODELS:
    cells = _split_cells(t)["default"]
    acc = {t: C.pct(sum(C.SELF[t][k] for k in cells), len(cells))}
    for p in MODELS:
        if p != t and (p, t) in C.CROSS:
            acc[p] = C.pct(sum(C.CROSS[(p, t)][k] for k in cells), len(cells))
    order = sorted(acc, key=acc.get)
    default_rank[t] = {
        "n_cells": len(cells),
        "accuracy": acc,
        "rank": {m: order.index(m) + 1 for m in acc},
        "lowest": order[0],
        "gap_lowest_to_next": round(acc[order[1]] - acc[order[0]], 1),
    }
RES["default_move_predictor_rank"] = default_rank


# Determined-cell self-advantage
# Completes the move-type split: the paired self-vs-best-other gap on determined (non-branch)
# cells, same methodology as the prior-aligned / idiosyncratic split above.

det = {}
for t in MODELS:
    bo = best_other_model(t)
    pairs = paired(
        C.SELF[t], C.CROSS[(bo, t)], restrict=lambda k, tt=t: not C.is_branch(tt, k[0], k[1])
    )
    ci = boot_gap_ci(pairs)
    det[t] = {**ci, **sign_flip_test(pairs), "best_other": bo}
RES["self_advantage_determined"] = det


# First vs later atypical cells (accumulated-error check)
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


# One-unvisited cells
# One-unvisited cells (2+ legal moves but a single unvisited one) are excluded from decision
# points: prediction there is far from ceiling, but no model shows a positive significant
# self-advantage, so the cells don't separate the models.

def _one_unvisited_cells(t):
    """Cells with two or more legal moves but exactly one unvisited."""
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


# Unique information (A3)
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
        p_vs[p] = C.fmt_p(raw_ps[p], N_PERM)
    cross_counts = {p: c for p, c in counts.items() if p != "self"}
    strongest = max(cross_counts, key=cross_counts.get)
    ci_lo, ci_hi = _boot_count_diff_ci(
        [(k[0], unique["self"][k], unique[strongest][k]) for k in keys]
    )
    unique_info[t] = {
        "n_cells": len(keys),
        "unique_correct": {(t if nm == "self" else nm): counts[nm] for nm in preds},
        "p_vs_each_cross": p_vs,
        "max_p": C.fmt_p(max(raw_ps.values()), N_PERM),
        "strongest_other": strongest,
        "count_diff_vs_strongest": counts["self"] - counts[strongest],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }
RES["unique_information"] = unique_info


# Atypical reasoning dependence (A5)
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
        "p_value_cluster_perm": C.fmt_p(_sign_flip_p(pairs), N_PERM) if n else None,
    }
RES["atypical_reasoning_dependence"] = reasoning_dep


# Run-to-run consistency by move type (D4)
# On the validation subsample, does run-to-run instability concentrate on atypical cells?
# Label-shuffle permutation of the atypical/other split against the per-cell agreement flag.
# Every row here is underpowered (single-digit unstable-cell counts); the emitted counts are
# the point, because the validation subsample can't resolve the question for any model.

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
        p = C.fmt_p(float((diffs >= obs - 1e-12).mean()), N_PERM)
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


# Atypical self-advantage on the alternate self-prediction runs
# Runs 1 and 2 are otherwise used only for the run-agreement noise floor; here all three runs
# are scored against the trajectory on the atypical cells that carry more than one run. That
# subset is a sample of the atypical cells rather than all of them, so its run-0 rate differs
# from the headline atypical rate, and both are emitted.

validation_runs = {}
for t in MODELS:
    cells = {}
    for mz, s, ri, rec in C.self_records(t, "reasoning"):
        if rec.get("parsed_position") is None:
            continue
        cells.setdefault((mz, s), {})[ri] = tuple(rec["parsed_position"])
    multi = {k: v for k, v in cells.items() if len(v) >= 2}
    atypical_cells = _split_cells(t)["atypical"]
    keys = [k for k in atypical_cells if k in multi]
    truth = {k: tuple(C.TRUTH[t][k[0]][k[1]]) for k in keys}
    cross_acc = {
        p: C.pct(sum(C.CROSS[(p, t)][k] for k in keys), len(keys)) if keys else None
        for p in MODELS
        if p != t and (p, t) in C.CROSS
    }
    self_acc, gaps = {}, {}
    for run in (0, 1, 2):
        scored = [k for k in keys if run in multi[k]]
        self_acc[run] = {
            "acc": C.pct(sum(multi[k][run] == truth[k] for k in scored), len(scored))
            if scored
            else None,
            "n": len(scored),
        }
        gaps[run] = {}
        for p in cross_acc:
            pr = [(k[0], multi[k][run] == truth[k], C.CROSS[(p, t)][k]) for k in scored]
            flip = sign_flip_test(pr)
            gaps[run][p] = {
                **boot_gap_ci(pr),
                "n_discordant": flip["n_discordant"],
                "p_value_cluster_perm": flip["p_value_cluster_perm"],
            }
    validation_runs[t] = {
        "n_cells": len(keys),
        "n_atypical_total": len(atypical_cells),
        "self_acc_by_run": self_acc,
        "cross_acc": cross_acc,
        "gap_by_run": gaps,
    }
RES["atypical_validation_runs"] = validation_runs


# Departing from a predictor's own generic answer
# The same paired contest as self_vs_best_other_paired, with the opponent replaced by the
# predictor's own modal answer about the other targets on the same cell. Both answers are
# scored against the predictor's own trajectory, so the gap is a like-for-like comparison and
# the existing clustered test applies unchanged. The cell tables themselves are in
# cross_structure.py; only the test lives here, because the analysis modules write on import
# and must not import one another.

MIN_DEPARTURES = 10  # below this neither the interval nor the p-value carries information


departures = {}
for t in MODELS:
    pairs = [
        (mz, self_ok, generic_ok)
        for mz, _, self_pred, generic, self_ok, generic_ok in C.generic_answer_cells(t)
        if self_pred != generic
    ]
    row = {**boot_gap_ci(pairs), **sign_flip_test(pairs)}
    if len(pairs) < MIN_DEPARTURES:
        row["ci_lo"] = row["ci_hi"] = row["p_value_cluster_perm"] = None
    departures[t] = row
RES["departure_payoff_test"] = departures


# Write

with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(RES, f, indent=1)
