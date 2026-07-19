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

import json
import os
import collections
import statistics as st

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
RES["regularity_vs_predictability"] = {
    "pearson_entropy_vs_predictability": pearson(_ent, _pr),
    "perm_p_entropy_vs_predictability": C.perm_corr_p(_ent, _pr),
    "pearson_firstlisted_vs_predictability": pearson(_fl, _pr),
    "perm_p_firstlisted_vs_predictability": C.perm_corr_p(_fl, _pr),
    "n_models": len(ms),
    "note": "negative entropy correlation / positive first-listed correlation => more rule-like models are more predictable; perm_p is a permutation test (n is only 5 models)",
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
# WRITE + SUMMARY
# ============================================================

with open(os.path.join(OUT, "exploration_strategy.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print("forced vs branch self-accuracy:")
    for t, d in forced_branch.items():
        print(
            f"  {t:7} forced={d['forced_self_acc']:5} (n={d['n_forced']:3})  branch={d['branch_self_acc']:5} (n={d['n_branch']:3})  drop={d['drop_at_branches']}"
        )
    print("\nbranch-choice regularity vs predictability:")
    print(f"  {'model':7} {'entropy':>8} {'first%':>7} {'backtr%':>8} {'predict':>8}")
    for t in MODELS:
        r = regularity[t]
        print(
            f"  {t:7} {str(r['direction_entropy']):>8} {str(r['first_listed_rate']):>7} {str(r['backtrack_rate_at_branches']):>8} {predictability[t]:>8}"
        )
    print(
        "  corr(entropy, predictability) =",
        RES["regularity_vs_predictability"]["pearson_entropy_vs_predictability"],
        f"(perm p={RES['regularity_vs_predictability']['perm_p_entropy_vs_predictability']})",
        "| corr(first-listed, predictability) =",
        RES["regularity_vs_predictability"]["pearson_firstlisted_vs_predictability"],
        f"(perm p={RES['regularity_vs_predictability']['perm_p_firstlisted_vs_predictability']})",
    )
    print("\nfirst-move: frac of step-1 that is a real branch, and step-1 self-acc:")
    for t, d in firstmove.items():
        print(
            f"  {t:7} branch={d['frac_step1_is_branch']:5}%  step1_acc={d['step1_self_acc']}  choices={d['step1_choice_distribution']}"
        )
    print("\ndeterminism: consistent mazes per model (multi-run):")
    for m, d in determinism.items():
        print(
            f"  {m:7} {d['n_consistent']:3}/{d['n_mazes']}  first-divergence-step hist: {d['first_divergence_step_hist']}"
        )
    print("\nbranch density by step (% of steps that are genuine choices):")
    for m, r in branch_rate_by_step.items():
        print(f"  {m:7} {r}")
    print("\nmid-horizon: Opus self-advantage vs branch density by step:")
    for row in midh["opus"]["by_step"]:
        print(
            f"  step {row['step']}: gap {row['gap']:+3} | self {row['self']:3} best_other {row['best_other']:3} prior {row['prior_nr']} | branch_rate {row['branch_rate']}"
        )
    print(f"  corr(gap, branch_rate) across steps = {midh['opus']['corr_gap_vs_branch_rate']}")
    print("\n-> wrote results/exploration_strategy.json")
