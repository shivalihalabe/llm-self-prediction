#!/usr/bin/env python3
"""
The no-reasoning prior
---

Characterizes where each model guesses the navigator ends up without working through the
maze, and how its reasoned predictions relate to that guess.

Measures:
- nr_accuracy_by_step: prior accuracy at step 1, and its decay over the horizon
- prior_concentration: entropy of the predicted-position distribution across mazes, where a
  low-entropy prior names nearly the same cell every maze
- cross_model_agreement: whether priors agree on a cell more than reasoned predictions do
- reasoning_correction: whether reasoning rescues a wrong prior, and whether a wrong
  reasoned answer lands on the prior's cell
- reasoned_consensus_vs_truth_vs_prior: whether the modal reasoned prediction tracks the
  truth or the prior
- nr_predicted_position_grid_by_step: where the prior lands on the grid

Output: analysis/results/prior.json
"""

import collections
import json
import os

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("prior")}


def _acc(scored, step=None):
    """Accuracy as a rounded float, or None."""
    v = C.acc(scored, None, step)[0]
    return round(v, 1) if v is not None else None


# Prior accuracy decays with horizon

acc_step = {}
for m in MODELS:
    acc_step[m] = {
        "nr": [_acc(C.SELF_NR[m], s) for s in range(1, 9)],
        "reasoning": [_acc(C.SELF[m], s) for s in range(1, 9)],
        "nr_overall": _acc(C.SELF_NR[m]),
        "reasoning_overall": _acc(C.SELF[m]),
    }
RES["nr_accuracy_by_step"] = acc_step


# Prior concentration (position entropy)

def _step_position_entropy(pos_dict):
    """mean over steps of the entropy of predicted positions across mazes (None-safe)."""
    per_step = []
    for s in range(1, 9):
        cells = collections.Counter(tuple(v) for (mz, st), v in pos_dict.items() if st == s)
        e = C.entropy(cells)
        per_step.append(round(e, 3) if e is not None else None)
    vals = [e for e in per_step if e is not None]
    return (round(sum(vals) / len(vals), 3) if vals else None), per_step


conc = {}
for m in MODELS:
    nr_mean, nr_steps = _step_position_entropy(C.SELF_NR_POS[m])
    r_mean, r_steps = _step_position_entropy(C.SELF_POS[m])
    conc[m] = {
        "nr_mean_position_entropy": nr_mean,
        "reasoning_mean_position_entropy": r_mean,
        "entropy_lift_from_reasoning": (
            round(r_mean - nr_mean, 3) if (nr_mean is not None and r_mean is not None) else None
        ),
        "per_step_nr": nr_steps,
        "per_step_reasoning": r_steps,
    }
RES["prior_concentration"] = conc


# Cross-model agreement, prior vs reasoned

def _cross_model_agreement(pos_dicts):
    """pairwise agreement on the predicted cell across models, on the shared 19-maze
    intersection."""
    by_step = {}
    agree = total = 0
    for mz in C.INTERSECTION:
        for s in range(1, 9):
            cells = [tuple(pos_dicts[m][(mz, s)]) for m in MODELS if (mz, s) in pos_dicts[m]]
            if len(cells) < 2:
                continue
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    total += 1
                    agree += cells[i] == cells[j]
                    d = by_step.setdefault(s, [0, 0])
                    d[0] += cells[i] == cells[j]
                    d[1] += 1
    return (
        C.pct(agree, total) if total else None,
        {s: C.pct(v[0], v[1]) for s, v in sorted(by_step.items())},
        total,
    )


nr_pct, nr_by_step, nr_n = _cross_model_agreement(C.SELF_NR_POS)
r_pct, r_by_step, r_n = _cross_model_agreement(C.SELF_POS)
RES["cross_model_agreement"] = {
    "nr_pairwise_agreement_pct": nr_pct,
    "reasoning_pairwise_agreement_pct": r_pct,
    "by_step_nr": nr_by_step,
    "by_step_reasoning": r_by_step,
    "n_pairs_nr": nr_n,
    "n_pairs_reasoning": r_n,
}


# Reasoning rescues vs gets trapped by the prior

correction = {}
for m in MODELS:
    keys = set(C.SELF[m]) & set(C.SELF_NR[m])
    r_correct = r_wrong = rescued = same_cell_when_wrong = both_present = agree = 0
    for k in keys:
        if k not in C.SELF_POS[m] or k not in C.SELF_NR_POS[m]:
            continue
        both_present += 1
        rp = tuple(C.SELF_POS[m][k])
        np_ = tuple(C.SELF_NR_POS[m][k])
        agree += rp == np_
        if C.SELF[m][k]:  # reasoning correct
            r_correct += 1
            if not C.SELF_NR[m][k]:  # but the prior was wrong -> reasoning rescued it
                rescued += 1
        else:  # reasoning wrong
            r_wrong += 1
            if rp == np_:  # and landed on the prior's cell -> trapped by prior
                same_cell_when_wrong += 1
    correction[m] = {
        "n_common": both_present,
        "r_correct_nr_wrong_pct": C.pct(rescued, r_correct) if r_correct else None,
        "r_wrong_on_prior_cell_pct": C.pct(same_cell_when_wrong, r_wrong) if r_wrong else None,
        "r_nr_prediction_agreement_pct": C.pct(agree, both_present) if both_present else None,
    }
RES["reasoning_correction"] = correction


# Reasoned consensus against truth and prior
# For each target, the modal reasoned prediction (self + all cross-predictors) at each (maze,step),
# compared to the target's truth and to its no-reasoning prior, by step. The self-prediction is
# first in the vote list and Counter.most_common preserves first-seen order, so a tied vote would
# resolve in favour of self; tied cells are excluded from both counts and reported as n_tied.

cons_track = {}
for t in MODELS:
    rows = []
    for s in range(1, 9):
        mt = mp = n = n_tied = 0
        for mz in C.CONSISTENT[t]:
            if (mz, s) not in C.SELF_POS[t] or (mz, s) not in C.SELF_NR_POS[t]:
                continue
            preds = [tuple(C.SELF_POS[t][(mz, s)])] + [
                tuple(C.CROSS_POS[(p, t)][(mz, s)])
                for p in MODELS
                if p != t and (p, t) in C.CROSS_POS and (mz, s) in C.CROSS_POS[(p, t)]
            ]
            if len(preds) < 2:
                continue
            counts = collections.Counter(preds).most_common()
            if len(counts) > 1 and counts[1][1] == counts[0][1]:
                n_tied += 1
                continue
            n += 1
            modal_r = counts[0][0]
            mt += modal_r == tuple(C.TRUTH[t][mz][s])
            mp += modal_r == tuple(C.SELF_NR_POS[t][(mz, s)])
        rows.append(
            {
                "step": s,
                "n": n,
                "n_tied": n_tied,
                "consensus_matches_truth_pct": C.pct(mt, n) if n else None,
                "consensus_matches_prior_pct": C.pct(mp, n) if n else None,
            }
        )
    cons_track[t] = rows
RES["reasoned_consensus_vs_truth_vs_prior"] = cons_track


# No-reasoning predicted-position grid by step
# Where no-reasoning self-predictions land on the 5x5 grid, pooled across models, per step:
# the spatial shape of the prior (concentration toward the centre / diagonal).

grids = {}
for step in range(1, 9):
    g = [[0] * C.COLS for _ in range(C.ROWS)]
    tot = 0
    for m in MODELS:
        for (mz, st2), pos in C.SELF_NR_POS[m].items():
            if st2 == step:
                g[pos[0]][pos[1]] += 1
                tot += 1
    grids[step] = {"grid": g, "tot": tot}
RES["nr_predicted_position_grid_by_step"] = grids


# Write

with open(os.path.join(OUT, "prior.json"), "w") as f:
    json.dump(RES, f, indent=1)
