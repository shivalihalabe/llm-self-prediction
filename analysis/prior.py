#!/usr/bin/env python3
"""
The no-reasoning prior
======================

The no-reasoning predictions are each model's *prior*: where it guesses the navigator ends up
without working through the maze. This script characterizes that prior and how reasoning relates
to it:

- nr_accuracy_by_step: the prior is fine at step 1 (few options) and collapses as the horizon grows.
- prior_concentration: entropy of the predicted-position distribution across mazes. A low-entropy
  prior predicts nearly the same cell every maze (a fixed template); reasoning should raise entropy
  by tracking each maze individually.
- cross_model_agreement: do different models' priors agree on the same cell more than their reasoned
  predictions do? If so, reasoning individuates models (each simulates its own path).
- reasoning_correction: when reasoning is right, did it rescue a wrong prior; when reasoning is wrong,
  did it stay trapped on the prior's cell.

Uses the no-reasoning scored/position dicts and reasoning dicts from common.py.
Output: analysis/results/prior.json
"""

import json
import os
import collections

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("prior")}


def _acc(scored, step=None):
    v = C.acc(scored, None, step)[0]
    return round(v, 1) if v is not None else None


# ============================================================
# PRIOR ACCURACY DECAYS WITH HORIZON
# ============================================================

acc_step = {}
for m in MODELS:
    acc_step[m] = {
        "nr": [_acc(C.SELF_NR[m], s) for s in range(1, 9)],
        "reasoning": [_acc(C.SELF[m], s) for s in range(1, 9)],
        "nr_overall": _acc(C.SELF_NR[m]),
        "reasoning_overall": _acc(C.SELF[m]),
    }
RES["nr_accuracy_by_step"] = acc_step


# ============================================================
# HOW FIXED IS THE PRIOR (position entropy)
# ============================================================


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


# ============================================================
# DOES REASONING INDIVIDUATE MODELS?
# ============================================================


def _cross_model_agreement(pos_dicts):
    """pairwise agreement on the predicted cell across models, on the shared 19-maze intersection."""
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
    "interpretation": "if NR agreement > reasoning agreement, reasoning individuates models (each tracks its own path)",
    "by_step_nr": nr_by_step,
    "by_step_reasoning": r_by_step,
    "n_pairs_nr": nr_n,
    "n_pairs_reasoning": r_n,
}


# ============================================================
# REASONING RESCUES VS GETS TRAPPED BY THE PRIOR
# ============================================================

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


# ============================================================
# DOES THE REASONED CONSENSUS TRACK TRUTH OR PRIOR?
# ============================================================

# For each target, the modal reasoned prediction (self + all cross-predictors) at each (maze,step),
# compared to the target's truth and to its no-reasoning prior, by step.
cons_track = {}
for t in MODELS:
    rows = []
    for s in range(1, 9):
        mt = mp = n = 0
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
            n += 1
            modal_r = collections.Counter(preds).most_common(1)[0][0]
            mt += modal_r == tuple(C.TRUTH[t][mz][s])
            mp += modal_r == tuple(C.SELF_NR_POS[t][(mz, s)])
        rows.append(
            {
                "step": s,
                "n": n,
                "consensus_matches_truth_pct": C.pct(mt, n) if n else None,
                "consensus_matches_prior_pct": C.pct(mp, n) if n else None,
            }
        )
    cons_track[t] = rows
RES["reasoned_consensus_vs_truth_vs_prior"] = cons_track


# ============================================================
# WRITE + SUMMARY
# ============================================================

with open(os.path.join(OUT, "prior.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print("no-reasoning prior accuracy by step (collapses as horizon grows):")
    print(f"  {'model':7} {'step1':>6} {'step2':>6} {'step4':>6} {'step8':>6}   {'overall':>8}")
    for m, d in acc_step.items():
        n = d["nr"]
        print(
            f"  {m:7} {str(n[0]):>6} {str(n[1]):>6} {str(n[3]):>6} {str(n[7]):>6}   {str(d['nr_overall']):>8}"
        )
    print("\nprior concentration - position entropy (NR vs reasoning); reasoning should be higher:")
    for m, d in conc.items():
        print(
            f"  {m:7} NR={d['nr_mean_position_entropy']}  reasoning={d['reasoning_mean_position_entropy']}  lift={d['entropy_lift_from_reasoning']}"
        )
    print("\ncross-model agreement on the predicted cell (19-maze intersection):")
    print(
        f"  NR priors agree {nr_pct}%  vs  reasoned predictions agree {r_pct}%  "
        f"-> reasoning {'individuates' if (nr_pct or 0) > (r_pct or 0) else 'does not individuate'} models"
    )
    print("\nreasoning vs the prior, per model:")
    for m, d in correction.items():
        print(
            f"  {m:7} when right, prior was wrong {d['r_correct_nr_wrong_pct']}% (rescued)  |  "
            f"when wrong, landed on prior cell {d['r_wrong_on_prior_cell_pct']}% (trapped)  |  R==NR {d['r_nr_prediction_agreement_pct']}%"
        )
    print("\nreasoned consensus vs truth vs prior, by step (opus):")
    for r in cons_track["opus"]:
        print(
            f"  step {r['step']}: matches_truth {r['consensus_matches_truth_pct']}%  matches_prior {r['consensus_matches_prior_pct']}%  (n={r['n']})"
        )
    print("\n-> wrote results/prior.json")
