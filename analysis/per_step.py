#!/usr/bin/env python3
"""
Horizon-resolved analyses
---

Collects the step-by-step views that don't belong to a single thematic script.

Measures:
- self_consistency_by_step: where over the horizon a model's runs stop agreeing
- determined_vs_branch_by_step: self-accuracy on determined against branch steps, per step
- self_matches_consensus_by_step: how far self tracks the consensus-of-others by step
- error_propagation: P(correct at k+1 | correct at k) against P(correct at k+1 | wrong at k)
- error_clustering_runs_test: whether correct and wrong steps cluster along a maze's row
- transition_legality: whether consecutive predicted positions form a legal walk
- predictability_horizon: the first step at which each target drops below 50%

Output: analysis/results/per_step.json
"""

import collections
import json
import os

import numpy as np
import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
N_DRAWS = 2000  # checkerboard draws for the error-clustering null
BURN_IN_SPANS = 10  # burn-in, in multiples of the matrix size
THIN_SPANS = 5  # draw interval, in multiples of the matrix size
SEED = 20260609
RES = {"metadata": C.metadata("per_step")}


# Self-consistency by step

def _consistency(m):
    """Per step, the fraction of validation cells whose runs all agree."""
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


# Determined vs branch accuracy by step
# Determined = two or more legal moves with exactly one unvisited (a forced step, by contrast,
# has a single legal move and is executed without an API call).

def _determined_vs_branch(m):
    """Per step, self-accuracy split into determined and branch cells."""
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
                "determined_acc": C.pct(f.mean()) if len(f) else None,
                "n_determined": int(len(f)),
                "branch_acc": C.pct(b.mean()) if len(b) else None,
                "n_branch": int(len(b)),
            }
        )
    return rows


RES["determined_vs_branch_by_step"] = {m: _determined_vs_branch(m) for m in MODELS}


# Self matches consensus-of-others by step

_ORDER = {p: i for i, p in enumerate(MODELS)}


def _consensus(votes):
    """Modal position; ties break by predictor order (Counter insertion order)."""
    return collections.Counter(votes).most_common(1)[0][0]


def consensus_by_step(t):
    """Per step, how often self matches the truth and the consensus of others."""
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


# Error propagation along the trajectory

def _propagation(m):
    """P(correct at k+1) conditioned on correctness at k, over consecutive steps."""
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


# Error clustering: adjacent agreement under a fixed-margin null
# Within a maze, do correct and wrong steps run together beyond what the maze's own
# difficulty and the horizon gradient already imply? Shuffling within each maze isn't enough
# on its own: accuracy falls with step, so a pure step gradient produces adjacent agreement
# by itself and would be scored as clustering. Holding both margins fixed, each maze's
# number of correct steps and each step's number of correct mazes, removes the maze effect
# and the gradient together and leaves only the run structure. Mixing was checked by doubling
# the draw interval, which moved the null mean by 0.16 points.

def _outcome_matrix(m):
    """Maze x step matrix, 1 where the run-0 self-prediction matched the trajectory."""
    mazes = sorted(C.CONSISTENT[m])
    grid = [[int(bool(C.SELF[m].get((mz, s)))) for s in range(1, 9)] for mz in mazes]
    return len(mazes), np.array(grid, dtype=np.int8)


def _adjacent_agreement(mat):
    """(matching adjacent step pairs, total adjacent step pairs) over a maze x step matrix."""
    same = mat[:, :-1] == mat[:, 1:]
    return int(same.sum()), int(same.size)


def _checkerboard_swaps(flat, n_cols, rows, cols, n):
    """Attempt n checkerboard swaps in place, which leaves both margins unchanged.

    Each attempt picks two rows and two columns; the 2x2 they span is swapped only when it
    reads [[1,0],[0,1]] or [[0,1],[1,0]], so every row and column sum is preserved.
    """
    for t in range(n):
        i, j = rows[2 * t], rows[2 * t + 1]
        if i == j:
            continue
        k, dst = cols[2 * t], cols[2 * t + 1]
        if k == dst:
            continue
        ik, il = i * n_cols + k, i * n_cols + dst
        jk, jl = j * n_cols + k, j * n_cols + dst
        v_ik, v_il = flat[ik], flat[il]
        if v_ik != v_il and v_ik == flat[jl] and v_il == flat[jk]:
            flat[ik], flat[il] = v_il, v_ik
            flat[jk], flat[jl] = v_ik, v_il


def _null_agreement_draws(mat):
    """Adjacent-agreement counts from a checkerboard chain started at the observed matrix."""
    n_rows, n_cols = mat.shape
    span = n_rows * n_cols
    flat = mat.flatten().tolist()
    rng = np.random.default_rng(SEED)

    def attempt(n):
        rows = rng.integers(0, n_rows, size=2 * n).tolist()
        cols = rng.integers(0, n_cols, size=2 * n).tolist()
        _checkerboard_swaps(flat, n_cols, rows, cols, n)

    attempt(BURN_IN_SPANS * span)
    draws = np.empty(N_DRAWS, dtype=np.int64)
    for d in range(N_DRAWS):
        attempt(THIN_SPANS * span)
        drawn = np.array(flat, dtype=np.int8).reshape(n_rows, n_cols)
        draws[d] = int((drawn[:, :-1] == drawn[:, 1:]).sum())
    return draws


def _clustering_row(observed, denom, draws, n_mazes):
    """One emitted row: the observed rate, the null it is measured against, and the p-value.

    The p-value adds one to both counts, so it cannot be zero; its floor is 1/(N_DRAWS + 1).
    """
    null = 100.0 * draws / denom
    return {
        "observed_pct": C.pct(observed, denom),
        "null_mean_pct": round(float(null.mean()), 1),
        "null_sd_pct": round(float(null.std(ddof=1)), 2),
        "p_value_fixed_margins": round(
            (int((draws >= observed).sum()) + 1) / (N_DRAWS + 1), 6
        ),
        "n_mazes": n_mazes,
        "n_pairs": denom,
    }


clustering = {}
pooled_obs = pooled_pairs = pooled_mazes = 0
pooled_draws = np.zeros(N_DRAWS, dtype=np.int64)
for m in MODELS:
    n_mazes, mat = _outcome_matrix(m)
    observed, denom = _adjacent_agreement(mat)
    draws = _null_agreement_draws(mat)
    clustering[m] = _clustering_row(observed, denom, draws, n_mazes)
    pooled_obs += observed
    pooled_pairs += denom
    pooled_mazes += n_mazes
    pooled_draws += draws
clustering["pooled"] = _clustering_row(pooled_obs, pooled_pairs, pooled_draws, pooled_mazes)
RES["error_clustering_runs_test"] = clustering


# Transition legality (route continuity)
# Do consecutive predicted positions chain into a legal walk? For each maze the sequence is
# (0,0) followed by the parsed run-0 predictions for steps 1-8; a pair is legal iff the two
# cells are Manhattan-adjacent with no wall between them. The null for the pair spanning
# steps k and k+1 is exact: with R_k the cells reachable from (0,0) by a walk of exactly k
# moves, the fraction of ordered pairs in R_k x R_{k+1} that are legal, averaged over mazes
# and then over pairs.

def _pair_legal(mz, a, b):
    """True if two cells are adjacent with no wall between them."""
    return C.manhattan(a, b) == 1 and frozenset([a, b]) not in C.WALLS[mz]


def _legality(pos_dict, target):
    """Legal-transition rates over a predictor's position sequences, overall and per step."""
    truth = C.TRUTH[target]
    n_pairs = n_legal = 0
    n_wrong = n_wrong_legal = n_corr = n_corr_legal = 0
    by_step = [[0, 0] for _ in range(8)]  # per consecutive-step pair (k -> k+1)
    for mz in sorted(C.CONSISTENT[target]):
        seq = {0: (0, 0)}
        for s in range(1, 9):
            if (mz, s) in pos_dict:
                seq[s] = tuple(pos_dict[(mz, s)])
        for k in range(8):
            if k not in seq or (k + 1) not in seq:
                continue
            a, b = seq[k], seq[k + 1]
            legal = _pair_legal(mz, a, b)
            n_pairs += 1
            n_legal += legal
            by_step[k][0] += legal
            by_step[k][1] += 1
            ok_a = a == tuple(truth[mz][k])
            ok_b = (k + 1) < len(truth[mz]) and b == tuple(truth[mz][k + 1])
            if ok_a and ok_b:
                n_corr += 1
                n_corr_legal += legal
            else:
                n_wrong += 1
                n_wrong_legal += legal
    return {
        "pct_legal": C.pct(n_legal, n_pairs),
        "n_pairs": n_pairs,
        "pct_legal_wrong_pairs": C.pct(n_wrong_legal, n_wrong),
        "n_wrong_pairs": n_wrong,
        "pct_legal_correct_pairs": C.pct(n_corr_legal, n_corr),
        "pct_legal_by_step": [C.pct(a2, b2) if b2 else None for a2, b2 in by_step],
    }


def _null_by_step(mazeset):
    """Exact chance legality for each step pair, averaged over a maze set."""
    per_step = []
    for k in range(8):
        fracs = []
        for mz in sorted(mazeset):
            r_k = C.reachable_exactly(mz, k)
            r_k1 = C.reachable_exactly(mz, k + 1)
            legal = sum(_pair_legal(mz, a, b) for a in r_k for b in r_k1)
            fracs.append(legal / (len(r_k) * len(r_k1)))
        per_step.append(100.0 * sum(fracs) / len(fracs))
    return per_step


legality = {"per_model": {}, "legality_matrix": {}}
# The scalar null averages over the model's own consistent set, like every accuracy in the
# repo; the by-step vector is over all 100 mazes, where it's a pure property of the maze
# set and identical for every model.
null_steps_all = _null_by_step(C.WALLS)
for m in MODELS:
    null_steps_own = _null_by_step(C.CONSISTENT[m])
    legality["per_model"][m] = {
        **_legality(C.SELF_POS[m], m),
        "null_pct": round(sum(null_steps_own) / len(null_steps_own), 1),
        "null_pct_by_step": [round(v, 1) for v in null_steps_all],
    }
for p2 in MODELS:
    legality["legality_matrix"][p2] = {}
    for t2 in MODELS:
        pos = C.SELF_POS[t2] if p2 == t2 else C.CROSS_POS.get((p2, t2))
        if pos is None:
            continue
        stats = _legality(pos, t2)
        legality["legality_matrix"][p2][t2] = {
            "pct_legal": stats["pct_legal"],
            "n_pairs": stats["n_pairs"],
        }
RES["transition_legality"] = legality


# Predictability horizon (first step below 50%)

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


# Write

with open(os.path.join(OUT, "per_step.json"), "w") as f:
    json.dump(RES, f, indent=1)
