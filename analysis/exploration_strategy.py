#!/usr/bin/env python3
"""
Navigation-side mechanism
=========================
What each model's exploration looks like, and why some
models are more predictable than others.

Covers forced vs branch self-accuracy, branch-choice regularity
(direction entropy + first-listed rate) correlated with target predictability, the
first-move puzzle, trajectory shape, and a determinism diagnosis (where the 3 nav runs
diverge for non-consistent mazes).

Operates on the run-0 navigation trajectories (common.TRUTH) plus the raw nav files for
the multi-run determinism check.

Output: analysis/results/exploration_strategy.json
"""

import collections
import json
import os
import statistics as st

import numpy as np
import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("exploration_strategy")}

direction = C.direction  # geometry/stat helpers are defined once in common.py
pearson = C.pearson
entropy = C.entropy


def _r3(x):
    return round(x, 3) if x is not None else None


# ============================================================
# BRANCH DECISION TABLE
# ============================================================
# One row per genuine choice (>=2 unvisited moves) in a run-0 trajectory: what was chosen,
# whether it was the first-listed legal direction, whether it backtracked. Everything
# regularity-related aggregates off this frame.


def _branch_rows():
    rows = []
    for t in MODELS:
        for mz in sorted(C.CONSISTENT[t]):
            traj = C.TRUTH[t][mz]
            for s in range(1, len(traj)):
                uvm = C.unvisited_moves(t, mz, s)
                if len(uvm) < 2:
                    continue
                a, b = tuple(traj[s - 1]), tuple(traj[s])
                ch = direction(a, b)
                legal_sorted = sorted(direction(a, nb) for nb in C.legal_moves(t, mz, s))
                rows.append((t, mz, s, ch, ch == legal_sorted[0], b not in [tuple(x) for x in uvm]))
    return pd.DataFrame(
        rows, columns=["target", "maze", "step", "chosen", "first_listed", "backtrack"]
    )


BRANCHES = _branch_rows()


# ============================================================
# FORCED VS BRANCH SELF-ACCURACY
# ============================================================


forced_branch = {}
for t in MODELS:
    f = [c for (mz, s), c in C.SELF[t].items() if len(C.unvisited_moves(t, mz, s)) == 1]
    b = [c for (mz, s), c in C.SELF[t].items() if len(C.unvisited_moves(t, mz, s)) >= 2]
    forced_branch[t] = {
        "forced_self_acc": C.pct(sum(f), len(f)) if f else None,
        "n_forced": len(f),
        "branch_self_acc": C.pct(sum(b), len(b)) if b else None,
        "n_branch": len(b),
        "drop_at_branches": (
            round(100.0 * sum(f) / len(f) - 100.0 * sum(b) / len(b), 1) if f and b else None
        ),
    }
RES["forced_vs_branch_self"] = forced_branch


# ============================================================
# BRANCH-CHOICE REGULARITY + PREDICTABILITY
# ============================================================


regularity = {}
predictability = {}
for t in MODELS:
    g = BRANCHES[BRANCHES.target == t]
    n_branch = len(g)
    chosen_dirs = collections.Counter(g.chosen)
    regularity[t] = {
        "n_branch_decisions": n_branch,
        "direction_entropy": _r3(entropy(chosen_dirs)),  # lower => more rule-like
        "first_listed_rate": C.pct(g.first_listed.mean()) if n_branch else None,
        "backtrack_rate_at_branches": C.pct(g.backtrack.mean()) if n_branch else None,
        "direction_distribution": dict(chosen_dirs),
    }
    # predictability of target t = mean native accuracy over all predictors incl self
    preds = [C.acc(C.SELF[t])[0]] + [
        C.acc(C.CROSS[(p, t)])[0] for p in MODELS if p != t and (p, t) in C.CROSS
    ]
    predictability[t] = round(st.mean([v for v in preds if v is not None]), 1)
RES["branch_choice_regularity"] = regularity
RES["target_predictability"] = predictability

# correlation across the 5 models: does branch regularity predict predictability?
ms = [m for m in MODELS if regularity[m]["direction_entropy"] is not None]
_ent = [regularity[m]["direction_entropy"] for m in ms]
_fl = [regularity[m]["first_listed_rate"] for m in ms]
_pr = [predictability[m] for m in ms]
# Rule-likeness under the corrected taxonomy: the default rate (share of decision points where
# the model took the alphabetically-first unvisited direction). The retired first-listed rate
# compressed the spread (35-81 vs 47.7-98.7) because refusals to backtrack counted as
# deviations.
_dr = []
for m in MODELS:
    dps = [
        (mz, s2)
        for mz in sorted(C.CONSISTENT[m])
        for s2 in range(1, len(C.TRUTH[m][mz]))
        if C.is_branch(m, mz, s2)
    ]
    _dr.append(C.pct(sum(C.chose_first_unvisited(m, mz, s2) for mz, s2 in dps), len(dps)))
RES["regularity_vs_predictability"] = {
    "pearson_default_rate_vs_predictability": pearson(_dr, _pr),
    "perm_p_default_rate_vs_predictability": C.perm_corr_p(_dr, _pr),
    "pearson_entropy_vs_predictability": pearson(_ent, _pr),
    "perm_p_entropy_vs_predictability": C.perm_corr_p(_ent, _pr),
    "n_models": len(ms),
    "note": "default rate = 100 - atypical rate under the first-unvisited taxonomy; "
    "positive default-rate correlation / negative entropy correlation => more rule-like "
    "models are more predictable; perm p is the seeded scipy two-sided convention used "
    "throughout (exact two-sided by |r| exceedance at n=5: 2/120 = 0.0167)",
}


# ============================================================
# FIRST-MOVE ANALYSIS (the step-1 puzzle)
# ============================================================


firstmove = {}
for t in MODELS:
    n = branch1 = 0
    choices = collections.Counter()
    for mz in sorted(C.CONSISTENT[t]):
        traj = C.TRUTH[t][mz]
        if len(traj) < 2:
            continue
        n += 1
        if len(C.unvisited_moves(t, mz, 1)) >= 2:
            branch1 += 1
        choices[direction(tuple(traj[0]), tuple(traj[1]))] += 1
    firstmove[t] = {
        "n_mazes": n,
        "frac_step1_is_branch": C.pct(branch1, n) if n else None,
        "step1_choice_distribution": dict(choices),
        "step1_self_acc": (
            round(C.acc(C.SELF[t], None, 1)[0], 1)
            if C.acc(C.SELF[t], None, 1)[0] is not None
            else None
        ),
    }
RES["first_move"] = firstmove


# ============================================================
# TRAJECTORY SHAPE
# ============================================================


shape = {}
for t in MODELS:
    cov, back, term = [], [], collections.Counter()
    for mz in sorted(C.CONSISTENT[t]):  # sorted: deterministic tie order in most_common
        traj = [tuple(p) for p in C.TRUTH[t][mz]]
        cov.append(len(set(traj)))
        back.append(len(traj) - len(set(traj)))  # revisited steps
        term[traj[-1]] += 1
    shape[t] = {
        "mean_unique_cells": round(st.mean(cov), 2),
        "mean_revisits": round(st.mean(back), 2),
        "top3_terminal_cells": [[list(p), c] for p, c in term.most_common(3)],
    }
RES["trajectory_shape"] = shape


# ============================================================
# DETERMINISM DIAGNOSIS (multi-run)
# ============================================================


determinism = {}
for m in MODELS:
    nav = json.load(open(os.path.join(C.DATA, "navigation", f"{m}_navigation.json")))["navigation"][
        m
    ]
    n_total = n_consistent = 0
    diverge_step = collections.Counter()
    for mz, obj in nav.items():
        runs = [r["trajectory"] for r in obj["runs"]]
        n_total += 1
        L = min(len(r) for r in runs)
        first = None
        for s in range(L):
            if len(set(tuple(r[s]) for r in runs)) > 1:
                first = s
                break
        if first is None and all(len(r) == len(runs[0]) for r in runs):
            n_consistent += 1
        else:
            diverge_step[first if first is not None else L] += 1
    determinism[m] = {
        "n_mazes": n_total,
        "n_consistent": n_consistent,
        "first_divergence_step_hist": dict(sorted(diverge_step.items())),
    }
RES["determinism_diagnosis"] = determinism


# ============================================================
# BRANCH DENSITY BY STEP + WHY THE MID-HORIZON
# ============================================================
# Where do genuine choices concentrate over the horizon, and does the self-advantage track them?


branch_rate_by_step = {}
for m in MODELS:
    rates = []
    for s in range(1, 9):
        n = tot = 0
        for mz in C.CONSISTENT[m]:
            if s < len(C.TRUTH[m][mz]):
                tot += 1
                n += C.is_branch(m, mz, s)
        rates.append(C.pct(n, tot) if tot else None)
    branch_rate_by_step[m] = rates
RES["branch_rate_by_step"] = branch_rate_by_step


def _best_other(t):
    c = {p: C.acc(C.CROSS[(p, t)])[0] for p in MODELS if p != t and (p, t) in C.CROSS}
    c = {p: v for p, v in c.items() if v is not None}
    return max(c, key=c.get) if c else None


midh = {}
for t in MODELS:
    bo = _best_other(t)
    if bo is None:
        continue
    rows, gaps, brs = [], [], []
    for i, s in enumerate(range(1, 9)):
        se = C.acc(C.SELF[t], None, s)[0]
        bb = C.acc(C.CROSS[(bo, t)], None, s)[0]
        pr = C.acc(C.SELF_NR[t], None, s)[0]
        br = branch_rate_by_step[t][i]
        if se is None or bb is None:
            continue
        rows.append(
            {
                "step": s,
                "self": round(se),
                "best_other": round(bb),
                "gap": round(se - bb),
                "prior_nr": round(pr) if pr is not None else None,
                "branch_rate": br,
            }
        )
        gaps.append(se - bb)
        if br is not None:
            brs.append(br)
    priors = [r["prior_nr"] for r in rows if r["prior_nr"] is not None]
    gaps_p = [r["gap"] for r in rows if r["prior_nr"] is not None]
    midh[t] = {
        "best_other": bo,
        "by_step": rows,
        "corr_gap_vs_branch_rate": C.pearson(brs, gaps) if len(brs) == len(gaps) else None,
        "corr_gap_vs_prior_nr": C.pearson(priors, gaps_p) if len(priors) >= 2 else None,
    }
RES["self_advantage_vs_branch_rate"] = midh


# ============================================================
# SELF-MODEL MISMATCH AT ATYPICAL CELLS
# ============================================================
# Why some models predict themselves worse than others do: at atypical cells, compare what the
# model actually chose against the first-listed move its simplified self-model would pick, where
# wrong self-predictions land, and how every predictor scores on exactly those cells.


mismatch = {}
for t in MODELS:
    cells = [
        (mz, s)
        for (mz, s) in C.SELF[t]
        if C.is_branch(t, mz, s) and not C.chose_first_unvisited(t, mz, s)
    ]
    chosen, firstlisted = collections.Counter(), collections.Counter()
    n_wrong = on_fl = 0
    for mz, s in cells:
        traj = [tuple(p) for p in C.TRUTH[t][mz]]
        a = traj[s - 1]
        chosen[direction(a, traj[s])] += 1
        unv = {direction(a, tuple(nb)): tuple(nb) for nb in C.unvisited_moves(t, mz, s)}
        fl = sorted(unv)[0]
        firstlisted[fl] += 1
        if C.SELF[t].get((mz, s)) is False:
            n_wrong += 1
            on_fl += tuple(C.SELF_POS[t][(mz, s)]) == unv[fl]
    # Permutation baseline for the first-listed landing rate: under the null, a wrong prediction
    # that lands on a legal non-chosen move is equally likely to land on any of them, so the
    # designated "first-listed" cell is no more likely than a random alternative. (The analogous
    # default-cell comparison is degenerate: there the first-listed cell IS the true cell, so a
    # wrong prediction can never land on it.)
    wrong = []
    for mz, s in cells:
        if C.SELF[t].get((mz, s)) is False:
            traj = [tuple(p2) for p2 in C.TRUTH[t][mz]]
            a = traj[s - 1]
            unv = {direction(a, tuple(nb)): tuple(nb) for nb in C.unvisited_moves(t, mz, s)}
            fl_cell = unv[sorted(unv)[0]]
            candidates = sorted(c2 for c2 in unv.values() if c2 != traj[s])
            wrong.append((tuple(C.SELF_POS[t][(mz, s)]), fl_cell, candidates, len(unv)))

    def _landing(records):
        """Landing test over the given wrong-prediction records: observed hits on the
        first-unvisited cell vs a null where a prediction landing on an unvisited
        non-chosen alternative is uniform over the alternatives."""
        on_alt = [w for w in records if w[0] in w[2]]
        expected = sum(1.0 / len(cand) for _, _, cand, _ in on_alt)
        observed = sum(pred == fl for pred, fl, _, _ in records)
        rng = np.random.default_rng(20260609)
        draws = np.zeros(10000, dtype=int)
        for pred, _, cand, _ in on_alt:
            draws += rng.integers(0, len(cand), 10000) == cand.index(pred) if pred in cand else 0
        return {
            "n_wrong": len(records),
            "on_firstlisted": observed,
            "pct": C.pct(observed, len(records)) if records else None,
            "landed_on_unvisited_nonchosen": len(on_alt),
            "expected_on_firstlisted_null": round(expected, 1),
            "expected_pct_null": C.pct(expected, len(records)) if records else None,
            "perm_p": round(float((draws >= observed).mean()), 4) if records else None,
        }

    # Pooled version kept for reference, labelled: at two-option cells the only alternative a
    # wrong prediction can land on IS the first-listed one, so observed and null are both ~100%
    # and those cells carry no information; pooling them understates the effect. The 3+ test is
    # the discriminating one: the tidy-rule hypothesis predicts one specific cell while random
    # error would spread across several.
    test_pooled = _landing(wrong)
    test_3plus = _landing([w for w in wrong if w[3] >= 3])
    mismatch[t] = {
        "firstlisted_landing_test_pooled": test_pooled,
        "firstlisted_landing_test_3plus": test_3plus,
        "atypical_cells_by_option_count": {
            "two_options": sum(1 for mz, s in cells if len(C.unvisited_moves(t, mz, s)) == 2),
            "three_plus_options": sum(
                1 for mz, s in cells if len(C.unvisited_moves(t, mz, s)) >= 3
            ),
        },
        "n_atypical_cells": len(cells),
        "chosen_direction_distribution": dict(chosen),
        "firstlisted_direction_distribution": dict(firstlisted),
        "wrong_self_pred_on_firstlisted_cell": {
            "n_wrong": n_wrong,
            "on_firstlisted": on_fl,
            "pct": C.pct(on_fl, n_wrong),
        },
    }
RES["self_model_mismatch"] = mismatch


# ============================================================
# PREDICTOR ACCURACY ON DEFAULT VS ATYPICAL CELLS (POOLED CROSS)
# ============================================================
# Cross-prediction only: accuracy on the target's default vs atypical cells.


pooled = {}
for p in MODELS:
    buckets = {"default": [], "atypical": []}
    for t in MODELS:
        if t == p:
            continue
        for (mz, step), okv in C.CROSS[(p, t)].items():
            if not C.is_branch(t, mz, step):
                continue
            key = "default" if C.chose_first_unvisited(t, mz, step) else "atypical"
            buckets[key].append(okv)
    pooled[p] = {k: {"acc": C.pct(sum(v), len(v)), "n": len(v)} for k, v in buckets.items()}
RES["predictor_default_vs_atypical_pooled"] = pooled


# ============================================================
# NO-BACKTRACKING UNANIMITY AND STEP-CATEGORY CENSUS (ALL 100 MAZES)
# ============================================================
# The empirical justification for the default/atypical taxonomy: at decision points no model
# ever took a visited direction, and with exactly one unvisited move every model took it --
# 3,578 opportunities, zero exceptions. The census gives the four-way breakdown of all 800
# run-0 steps per model, over all 100 mazes.


def _cell(t, mz, step):
    traj = [tuple(p) for p in C.TRUTH[t][mz]]
    pos = traj[step - 1]
    legal = {direction(pos, tuple(nb)): tuple(nb) for nb in C.legal_moves(t, mz, step)}
    vis = set(traj[:step])
    unv = {d: c for d, c in legal.items() if c not in vis}
    return legal, unv, direction(pos, traj[step])


census = {}
backtrack = {}
for t in MODELS:
    counts = {"forced": 0, "zero_unvisited": 0, "one_unvisited": 0, "decision_points": 0}
    zero_took_first = dp_took_visited = one_took_unvisited = 0
    for mz in C.TRUTH[t]:
        for step in range(1, len(C.TRUTH[t][mz])):
            legal, unv, ch = _cell(t, mz, step)
            if len(legal) == 1:
                counts["forced"] += 1
            elif len(unv) == 0:
                counts["zero_unvisited"] += 1
                zero_took_first += ch == sorted(legal)[0]
            elif len(unv) == 1:
                counts["one_unvisited"] += 1
                one_took_unvisited += ch in unv
            else:
                counts["decision_points"] += 1
                dp_took_visited += ch not in unv
    census[t] = {**counts, "zero_unvisited_took_first_listed": zero_took_first}
    backtrack[t] = {
        "dp_took_visited_direction": {"count": dp_took_visited, "n": counts["decision_points"]},
        "one_unvisited_took_it": {"count": one_took_unvisited, "n": counts["one_unvisited"]},
    }
backtrack["total"] = {
    k: {
        "count": sum(backtrack[m][k]["count"] for m in MODELS),
        "n": sum(backtrack[m][k]["n"] for m in MODELS),
    }
    for k in ("dp_took_visited_direction", "one_unvisited_took_it")
}
RES["step_category_census"] = census
RES["no_backtracking"] = backtrack


# ============================================================
# DEVIATION PROFILE (NEW TAXONOMY) AND RULE-LIKENESS VS PREDICTABILITY
# ============================================================
# Within each consistent set: decision points, default vs atypical under the first-unvisited
# rule, the atypical rate (the sharper rule-likeness measure), and the South share of atypical
# moves. The correlation re-tests regularity -> predictability with the new measure.


profile = {}
for t in MODELS:
    dp = default = 0
    atyp_dirs = []
    for mz in sorted(C.CONSISTENT[t]):
        for step in range(1, len(C.TRUTH[t][mz])):
            if not C.is_branch(t, mz, step):
                continue
            dp += 1
            _, _, ch = _cell(t, mz, step)
            if C.chose_first_unvisited(t, mz, step):
                default += 1
            else:
                atyp_dirs.append(ch)
    profile[t] = {
        "n_mazes": len(C.CONSISTENT[t]),
        "decision_points": dp,
        "default": default,
        "atypical": dp - default,
        "atypical_rate": C.pct(dp - default, dp),
        "south_share_of_atypical": C.pct(sum(d == "South" for d in atyp_dirs), len(atyp_dirs)),
    }
RES["deviation_profile"] = profile


# ============================================================
# WRITE
# ============================================================


with open(os.path.join(OUT, "exploration_strategy.json"), "w") as f:
    json.dump(RES, f, indent=1)
