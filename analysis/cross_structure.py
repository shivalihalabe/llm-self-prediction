#!/usr/bin/env python3
"""
Cross-prediction structure
---

Relates the five predictors of each target to one another, then aggregates across targets.
Reads the predicted-coordinate dicts, common.SELF_POS and common.CROSS_POS.

Measures:
- convergence: the spread of the five predictor accuracies on one target
- position agreement: whether predictors pick the same coordinate, not just both-right
- ensemble: whether a majority vote beats the best single predictor
- wrong-answer agreement: whether jointly wrong predictors name the same wrong cell,
  against the rate two independent wrong guesses would coincide at
- self against consensus: whether self's prediction resembles the truth or the others' vote
- predictor ranges: the row spread, and the spread with each predictor left out
- specialization: a joint additive fit over the twenty cross cells, and its residuals
- developer affinity: whether same-developer pairs carry a higher residual, on n=2 pairs
- self-projection: whether a predictor does better on targets that behave like it
- asymmetry: the A->B minus B->A gap against target predictability
- target tracking: cross-accuracy against the target's own self-accuracy
- predictor invariance: whether one predictor's answer changes with the target named,
  against the rate independent answers would coincide at
- the generic answer: whether a predictor's answer about itself is just its modal
  answer about the others, and whether departing from it helps
- held-out agreement: whether an answer about itself sits closer to the rest of its cell
  than an answer about another model does

Output: analysis/results/cross_structure.json
"""

import collections
import itertools
import json
import os
import statistics as st

import numpy as np
import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("cross_structure")}


pearson = C.pearson  # defined once in common.py


def predictors_for(t):
    """{name: position_dict} for target t: self + every cross predictor."""
    d = {"self": C.SELF_POS[t]}
    for p in MODELS:
        if p != t and (p, t) in C.CROSS_POS:
            d[p] = C.CROSS_POS[(p, t)]
    return d


def wide_positions(t):
    """One row per (maze, step) every predictor answered; columns = predictors + truth (tuples)."""
    preds = predictors_for(t)
    frames = [pd.Series({k: tuple(v) for k, v in d.items()}, name=nm) for nm, d in preds.items()]
    wide = pd.concat(frames, axis=1, join="inner")
    wide["truth"] = [tuple(C.TRUTH[t][mz][s]) for mz, s in wide.index]
    return wide


# Per-target structure

struct = {}
for t in MODELS:
    wide = wide_positions(t)
    names = [c for c in wide.columns if c != "truth"]
    others = [nm for nm in names if nm != "self"]
    n = len(wide)

    correct = wide[names].eq(wide["truth"], axis=0)
    per_pred_acc = {nm: C.pct(correct[nm].mean()) if n else None for nm in names}

    # convergence: spread of the predictor accuracies
    accs = [v for v in per_pred_acc.values() if v is not None]
    convergence = {
        "std": round(st.pstdev(accs), 2) if len(accs) > 1 else None,
        "range": round(max(accs) - min(accs), 1) if accs else None,
    }

    # Chance baseline for the jointly-wrong agreement below. Two wrong answers drawn
    # independently and uniformly from the positions a walk of exactly s steps can reach,
    # minus the true one, coincide with probability 1/k, so the null needs no simulation. It
    # is accumulated per cell rather than per target, because k changes with the step and the
    # jointly-wrong cells aren't spread evenly across steps. Where only the true position is
    # reachable there is no pool; every predictor names it, so no jointly-wrong pair falls in
    # such a cell and the observed rate and the baseline share one denominator.
    pool = pd.Series(
        [len(C.reachable_exactly(mz, s) - {tr}) for (mz, s), tr in zip(wide.index, wide.truth)],
        index=wide.index,
    )
    coincide = pool.map(lambda k: 1.0 / k if k else 0.0)

    # inter-predictor position agreement (all pairs) + agreement among the jointly-wrong
    pair_agree = pair_total = wrong_pair_agree = wrong_pair_total = 0
    wrong_pair_chance = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            same = wide[a].eq(wide[b])
            pair_agree += int(same.sum())
            pair_total += n
            both_wrong = ~correct[a] & ~correct[b]
            wrong_pair_agree += int((same & both_wrong).sum())
            wrong_pair_total += int(both_wrong.sum())
            wrong_pair_chance += float(coincide[both_wrong].sum())

    # ensemble majority vote (ties break by predictor order, as in Counter insertion)
    mv_others = wide[others].apply(lambda r: collections.Counter(r).most_common(1)[0][0], axis=1)
    mv_all = wide[names].apply(lambda r: collections.Counter(r).most_common(1)[0][0], axis=1)
    # tie flag for the others-vote: the top count is shared by two or more positions
    tie_others = wide[others].apply(
        lambda r: (lambda mc: len(mc) > 1 and mc[1][1] == mc[0][1])(
            collections.Counter(r).most_common()
        ),
        axis=1,
    )

    struct[t] = {
        "n_shared_items": n,
        "per_predictor_acc": per_pred_acc,
        "convergence": convergence,
        "mean_pairwise_position_agreement_pct": (
            C.pct(pair_agree, pair_total) if pair_total else None
        ),
        "wrong_predictor_agreement_pct": (
            C.pct(wrong_pair_agree, wrong_pair_total) if wrong_pair_total else None
        ),
        "n_wrong_pairs": wrong_pair_total,
        "wrong_pair_chance_pct": (
            C.pct(wrong_pair_chance, wrong_pair_total) if wrong_pair_total else None
        ),
        "n_cells_single_endpoint": int((pool == 0).sum()),
        "best_single_acc": max(accs) if accs else None,
        "ensemble_others_acc": C.pct(mv_others.eq(wide["truth"]).mean()) if n else None,
        "ensemble_all_acc": C.pct(mv_all.eq(wide["truth"]).mean()) if n else None,
        "self_matches_truth_pct": C.pct(correct["self"].mean()) if n else None,
        "self_matches_consensus_pct": C.pct(wide["self"].eq(mv_others).mean()) if n else None,
        "consensus_n_tied": int(tie_others.sum()),
        "self_matches_consensus_excl_ties_pct": (
            C.pct(wide["self"][~tie_others].eq(mv_others[~tie_others]).mean())
            if int((~tie_others).sum())
            else None
        ),
    }
RES["per_target_structure"] = struct


# Per-target predictor ranges (leave-one-out)
# For each target, the spread of the five predictor accuracies and the spread recomputed with
# each predictor removed in turn, so the predictor accounting for a wide row is identified by
# the data rather than assumed.

ranges = {}
for t in MODELS:
    accs = {
        (t if nm == "self" else nm): v for nm, v in struct[t]["per_predictor_acc"].items()
    }
    ranges[t] = {
        "all_predictors": accs,
        "range_all": round(max(accs.values()) - min(accs.values()), 1),
        "range_without": {
            m: round(
                max(v for nm, v in accs.items() if nm != m)
                - min(v for nm, v in accs.items() if nm != m),
                1,
            )
            for m in accs
        },
    }
RES["per_target_predictor_ranges"] = ranges


# Asymmetry tracks predictability
# For each unordered pair, (A->B minus B->A) should track (predictability_B minus
# predictability_A). Each pair's predictability difference excludes that pair's own two
# directions: predictability[b] would otherwise contain acc(a->b), which is the y variable, so
# x would contain exactly y/5 by construction and the correlation would be partly mechanical.
# The ten pairs are still not independent (each model appears in four of them), the same
# non-independence recorded against cross_acc_vs_target_self_acc below.

def _acc_of(p2, t2):
    """Accuracy of predictor p2 on target t2, using the self cell on the diagonal."""
    return C.acc(C.SELF[t2] if p2 == t2 else C.CROSS[(p2, t2)])[0]


acc_gaps, pred_gaps, pairs_dump = [], [], []
for i in range(len(MODELS)):
    for j in range(i + 1, len(MODELS)):
        a, b = MODELS[i], MODELS[j]
        if (a, b) not in C.CROSS or (b, a) not in C.CROSS:
            continue
        ab = C.acc(C.CROSS[(a, b)])[0]
        ba = C.acc(C.CROSS[(b, a)])[0]
        if ab is None or ba is None:
            continue
        pred_b = st.mean(_acc_of(p2, b) for p2 in MODELS if p2 != a)
        pred_a = st.mean(_acc_of(p2, a) for p2 in MODELS if p2 != b)
        acc_gaps.append(ab - ba)
        pred_gaps.append(pred_b - pred_a)
        pairs_dump.append(
            {
                "pair": f"{a}|{b}",
                "a_to_b": round(ab, 1),
                "b_to_a": round(ba, 1),
                "acc_gap": round(ab - ba, 1),
                "predictability_gap_b_minus_a": round(pred_b - pred_a, 1),
            }
        )
# perm_p is the seeded scipy two-sided convention (doubled tail), as at every perm_corr_p site.
RES["asymmetry_vs_predictability"] = {
    "pearson": pearson(pred_gaps, acc_gaps),
    "perm_p": C.perm_corr_p(pred_gaps, acc_gaps),
    "n_pairs": len(pred_gaps),
    "pairs": pairs_dump,
}


# Cross-accuracy tracks target self-accuracy


# The x variable (target self-accuracy) takes only five distinct values, one per target, so
# the twenty cells aren't exchangeable; the permutation relabels at the target level and is
# exhaustive over all 120 orderings. Unlike the scipy convention used elsewhere, the p-value
# here counts |r|-exceedance directly (identity relabelling included), so its floor is 1/120.
self_acc = {t: C.acc(C.SELF[t])[0] for t in MODELS}
cells_by_target, ys = [], []
for p in MODELS:
    for t in MODELS:
        if t == p or (p, t) not in C.CROSS:
            continue
        cr = C.acc(C.CROSS[(p, t)])[0]
        if cr is not None and self_acc[t] is not None:
            cells_by_target.append(t)
            ys.append(cr)
xs = [self_acc[t] for t in cells_by_target]
r_obs = np.corrcoef(xs, ys)[0, 1]
n_exceed = 0
for perm in itertools.permutations(MODELS):
    relabel = dict(zip(MODELS, perm))
    xp = [self_acc[relabel[t]] for t in cells_by_target]
    n_exceed += abs(np.corrcoef(xp, ys)[0, 1]) >= abs(r_obs) - 1e-12
n_perms = 120
RES["cross_acc_vs_target_self_acc"] = {
    "pearson": pearson(xs, ys),
    "perm_p": round(int(n_exceed) / n_perms, 6),
    "n_permutations": n_perms,
    "n_cells": len(xs),
}


# Oracle ceiling + self's unique contribution

oracle = {}
for t in MODELS:
    wide = wide_positions(t)
    names = [c for c in wide.columns if c != "truth"]
    others = [nm for nm in names if nm != "self"]
    n = len(wide)
    correct = wide[names].eq(wide["truth"], axis=0)
    any_other = correct[others].any(axis=1)
    oracle[t] = {
        "n": n,
        "best_single_acc": struct[t]["best_single_acc"],
        "oracle_any_other_acc": C.pct(any_other.mean()) if n else None,
        "oracle_any_incl_self_acc": C.pct((any_other | correct["self"]).mean()) if n else None,
        "items_only_self_correct_pct": C.pct((correct["self"] & ~any_other).mean()) if n else None,
    }
RES["oracle_ceiling"] = oracle


# Predictor x target specialization
# Additive model acc ~ mu + a[predictor] + b[target], fit jointly by least squares over the
# 20 cross cells with sum-to-zero constraints on both effect vectors. Subtracting marginal
# means instead is degenerate on this diagonal-free design: the mean residual per predictor
# reduces to the grand mean minus the mean difficulty of that predictor's target set (and the
# per-target mirror reduces the same way), carrying no information about the predictor. Do
# not add either marginal mean back.

cells = {
    (p, t): C.acc(C.CROSS[(p, t)])[0]
    for p in MODELS
    for t in MODELS
    if t != p and (p, t) in C.CROSS
}
# Sum-to-zero enforced by eliminating the last effect in each vector: a[last] = -sum(rest).
_rows, _rhs = [], []
for (p, t), v in cells.items():
    row = np.zeros(1 + 2 * (len(MODELS) - 1))
    row[0] = 1.0
    pi, ti = MODELS.index(p), MODELS.index(t)
    if pi < len(MODELS) - 1:
        row[1 + pi] = 1.0
    else:
        row[1 : len(MODELS)] = -1.0
    if ti < len(MODELS) - 1:
        row[len(MODELS) + ti] = 1.0
    else:
        row[len(MODELS) :] = -1.0
    _rows.append(row)
    _rhs.append(v)
_theta = np.linalg.lstsq(np.array(_rows), np.array(_rhs), rcond=None)[0]
mu = float(_theta[0])
a_eff = list(_theta[1 : len(MODELS)]) + [-float(sum(_theta[1 : len(MODELS)]))]
b_eff = list(_theta[len(MODELS) :]) + [-float(sum(_theta[len(MODELS) :]))]
pred_effect = {m: float(v) for m, v in zip(MODELS, a_eff)}
targ_effect = {m: float(v) for m, v in zip(MODELS, b_eff)}
resid_raw = {
    f"{p}->{t}": v - (mu + pred_effect[p] + targ_effect[t]) for (p, t), v in cells.items()
}
resid = {k: round(v, 1) for k, v in resid_raw.items()}
RES["predictor_target_specialization"] = {
    "mu": round(mu, 1),
    "predictor_effect": {m: round(v, 1) for m, v in pred_effect.items()},
    "target_effect": {m: round(v, 1) for m, v in targ_effect.items()},
    "residuals": resid,
    "note": "joint least-squares fit acc ~ mu + a[predictor] + b[target] over the 20 cross "
    "cells, sum-to-zero effects; residual = acc - (mu + a[predictor] + b[target])",
}


# Developer affinity (same developer vs different)
# Developers: anthropic={opus,sonnet}, openai={gpt}, zhipu={glm}, alibaba={qwen}. The only
# same-developer pair is opus<->sonnet, so the test is underpowered at n=2.

DEVELOPER = {
    "opus": "anthropic",
    "sonnet": "anthropic",
    "gpt": "openai",
    "glm": "zhipu",
    "qwen": "alibaba",
}
same_dev = sorted(k for k in resid if DEVELOPER[k.split("->")[0]] == DEVELOPER[k.split("->")[1]])
diff_dev = [k for k in resid if DEVELOPER[k.split("->")[0]] != DEVELOPER[k.split("->")[1]]]
RES["developer_affinity"] = {
    "same_developer_pairs": {k: resid[k] for k in same_dev},
    "mean_residual_same_developer": (
        round(st.mean([resid_raw[k] for k in same_dev]), 2) if same_dev else None
    ),
    "mean_residual_different_developer": (
        round(st.mean([resid_raw[k] for k in diff_dev]), 2) if diff_dev else None
    ),
    "note": "developers: anthropic={opus,sonnet}, openai={gpt}, zhipu={glm}, alibaba={qwen}. "
    "Only opus<->sonnet is same-developer (n=2, asymmetric) so this is underpowered.",
}


# Self vs other-prediction dissociation
# Is a model good at predicting itself, at predicting others, both, or neither?
# (Opus vs Sonnet differ sharply.)


# self_advantage_adjusted subtracts the joint-fit prediction of what predictor m would score
# on its own cells (mu + a[m] + b[m]), removing both the model's simulator skill and its own
# predictability; predictability_adjusted is the fit's target margin (mu + b[m]), free of the
# which-four-predictors composition that skews the raw target mean. Both are point estimates
# from a twenty-cell fit with no uncertainty attached; the inferential weight for
# self-advantage stays with the clustered paired tests in stats.py.
dissociation = {}
for m in MODELS:
    self_acc = C.acc(C.SELF[m])[0]
    as_predictor = [C.acc(C.CROSS[(m, t)])[0] for t in MODELS if t != m and (m, t) in C.CROSS]
    as_predictor = [v for v in as_predictor if v is not None]
    as_target = [C.acc(C.CROSS[(p, m)])[0] for p in MODELS if p != m and (p, m) in C.CROSS]
    as_target = [v for v in as_target if v is not None]
    dissociation[m] = {
        "self_acc": round(self_acc, 1),
        "mean_acc_predicting_others": round(st.mean(as_predictor), 1) if as_predictor else None,
        "predictability_by_others": round(st.mean(as_target), 1) if as_target else None,
        "predictability_adjusted": round(mu + targ_effect[m], 1),
        "self_advantage_adjusted": round(self_acc - (mu + pred_effect[m] + targ_effect[m]), 1),
    }
RES["self_vs_other_prediction_dissociation"] = dissociation


# Cross-prediction as self-projection
# Does a predictor predict a target better when the target behaves like the predictor?
# traj_similarity is symmetric, so the twenty ordered pairs carry only ten distinct x values while
# the residuals are asymmetric, and the permutation treats them as exchangeable.

def traj_similarity(p, t):
    """Percent of shared-maze steps where two models occupy the same cell."""
    shared = C.CONSISTENT[p] & C.CONSISTENT[t]
    n = agree = 0
    for mz in shared:
        L = min(len(C.TRUTH[p][mz]), len(C.TRUTH[t][mz]))
        for s in range(1, L):
            n += 1
            agree += tuple(C.TRUTH[p][mz][s]) == tuple(C.TRUTH[t][mz][s])
    return (100.0 * agree / n) if n else None


sims, accs, resids, dump = [], [], [], []
for p in MODELS:
    for t in MODELS:
        if p == t or (p, t) not in C.CROSS:
            continue
        sim = traj_similarity(p, t)
        cr = C.acc(C.CROSS[(p, t)])[0]
        if sim is None or cr is None:
            continue
        sims.append(sim)
        accs.append(cr)
        resids.append(resid_raw[f"{p}->{t}"])
        dump.append(
            {
                "pair": f"{p}->{t}",
                "behavioral_similarity": round(sim, 1),
                "cross_acc": round(cr, 1),
                "residual": resid[f"{p}->{t}"],
            }
        )
RES["self_projection"] = {
    "pearson_similarity_vs_cross_acc": C.pearson(sims, accs),
    "perm_p_raw": C.perm_corr_p(sims, accs),
    "pearson_similarity_vs_residual": C.pearson(
        sims, resids
    ),  # controls for target/predictor main effects
    "perm_p_residual": C.perm_corr_p(sims, resids),
    "pairs": sorted(dump, key=lambda d: d["residual"], reverse=True),
}


# Pairwise behavioral similarity matrix
# The trajectory-overlap matrix behind the self-projection correlation: traj_similarity for
# every ordered pair (pooled per-step position match over shared consistent mazes, excluding
# the shared start cell).

RES["behavioral_similarity_matrix"] = {
    a: {b: round(traj_similarity(a, b), 1) for b in MODELS if b != a} for a in MODELS
}


# Step agreement on all 100 mazes and oracle composition by move type
# step_agreement_all_mazes: fraction of run-0 steps where two models occupy the identical
# position, over all 100 mazes (behavioral_similarity_matrix above is the shared consistent-set
# version). oracle_composition: what share of each model's only-self-correct cells (self right,
# all four cross-predictors wrong, on cells every predictor answered) are atypical / default /
# determined under the first-unvisited taxonomy.

agree_all = {}
for a in MODELS:
    agree_all[a] = {}
    for b in MODELS:
        if a == b:
            continue
        n = ok = 0
        for mz in C.TRUTH[a]:
            ta, tb = C.TRUTH[a][mz], C.TRUTH[b][mz]
            L = min(len(ta), len(tb))
            for step in range(1, L):
                n += 1
                ok += tuple(ta[step]) == tuple(tb[step])
        agree_all[a][b] = C.pct(ok, n)
RES["step_agreement_all_mazes"] = agree_all

composition = {}
for t in MODELS:
    kinds = {"atypical": 0, "default": 0, "determined": 0}
    for mz, step in sorted(C.SELF[t]):
        others = [C.CROSS[(p, t)].get((mz, step)) for p in MODELS if p != t]
        if any(o is None for o in others):
            continue
        if not (C.SELF[t][(mz, step)] and not any(others)):
            continue
        if not C.is_branch(t, mz, step):
            kinds["determined"] += 1
        elif C.chose_first_unvisited(t, mz, step):
            kinds["default"] += 1
        else:
            kinds["atypical"] += 1
    tot = sum(kinds.values())
    composition[t] = {
        **kinds,
        "n": tot,
        "atypical_share_pct": C.pct(kinds["atypical"], tot) if tot else None,
    }
RES["oracle_composition_by_move_type"] = composition


# Predictor invariance across targets
# Every measure above compares different predictors of one target. This asks the reverse:
# when a predictor is asked about several targets on the same maze and step, does its answer
# change? An answer that never changes is one simulation of the maze wearing whichever name
# the prompt supplied, the predictor's own included. A cell's rows are the targets whose
# consistent set holds that maze, so a cell with one row says nothing about invariance and
# is dropped rather than counted as invariant.

def _predictor_cells(rows):
    """Group prediction rows into {(predictor, maze, step): [row, ...]}."""
    out = collections.defaultdict(list)
    for r in rows.itertuples(index=False):
        out[(r.predictor, r.maze, r.step)].append(r)
    return out


def _invariance(grouped):
    """Per predictor and pooled: cells holding two or more targets, and the share of those
    where every target drew the same predicted position."""
    out = {}
    pooled_n = pooled_inv = 0
    for p in MODELS:
        multi = [v for k, v in grouped.items() if k[0] == p and len(v) >= 2]
        inv = sum(1 for v in multi if len({r.pred for r in v}) == 1)
        out[p] = {
            "n_cells": len(multi),
            "n_invariant": inv,
            "invariant_pct": C.pct(inv, len(multi)),
        }
        pooled_n += len(multi)
        pooled_inv += inv
    out["pooled"] = {
        "n_cells": pooled_n,
        "n_invariant": pooled_inv,
        "invariant_pct": C.pct(pooled_inv, pooled_n),
    }
    # Chance floor for the rates above. If the m answers in a cell were independent uniform
    # draws from the k positions a walk of exactly that many steps can reach, all m coincide
    # with probability k ** (1 - m), so the floor is closed-form and nothing is simulated. It
    # is emitted once per variant, not per predictor: every predictor sees the same cells, so
    # five copies of one number would invite a comparison that isn't there. Unlike
    # wrong_pair_chance_pct in the per-target block, the true position stays in the pool,
    # since this asks how often the answers agree rather than how often two wrong ones do.
    # It is a mark to read the bars against, not a null anyone is testing.
    floors = [
        len(C.reachable_exactly(mz, step)) ** (1 - len(rows))
        for (owner, mz, step), rows in grouped.items()
        if owner == MODELS[0] and len(rows) >= 2
    ]
    out["chance_invariant_pct"] = C.pct(sum(floors), len(floors))
    return out


scored_rows = C.RECORDS[C.RECORDS.kind.isin(("self", "cross"))]
# branch status belongs to the target's own route, so rows are filtered before they are
# grouped; filtering afterwards would keep rows whose target never faced a choice there
branch_rows = scored_rows[
    [C.is_branch(r.target, r.maze, r.step) for r in scored_rows.itertuples(index=False)]
]
all_cells = _predictor_cells(scored_rows)
invariance = {
    "all_steps": _invariance(all_cells),
    "decision_points": _invariance(_predictor_cells(branch_rows)),
}
# a cell's row count is the number of models consistent on that maze, which doesn't depend on
# the predictor, so every predictor must see the same number of cells; a mismatch means the
# scoping is wrong
for variant in invariance.values():
    sizes = {v["n_cells"] for k, v in variant.items() if k in MODELS}
    assert len(sizes) == 1, f"predictors see different cell counts: {sizes}"
RES["predictor_invariance"] = invariance


# The generic answer, and what departing from it costs
# The generic answer is the modal position a predictor gave about the other targets in a
# cell. Comparing the predictor's answer about itself against it separates a genuine
# self-model from one route relabelled. Where the two differ, both are scored against the
# predictor's own trajectory, so the accuracies are directly comparable; stats.py tests that
# gap.

same_as_generic = {}
payoff = {}
for p in MODELS:
    cells = C.generic_answer_cells(p)
    matched = sum(1 for _, _, self_pred, generic, _, _ in cells if self_pred == generic)
    same_as_generic[p] = {
        "n_cells": len(cells),
        "n_equal": matched,
        "self_equals_generic_pct": C.pct(matched, len(cells)),
    }
    departures = [(sc, gc) for _, _, self_pred, generic, sc, gc in cells if self_pred != generic]
    n_self = sum(1 for sc, _ in departures if sc)
    n_generic = sum(1 for _, gc in departures if gc)
    payoff[p] = {
        "n_departures": len(departures),
        "n_self_correct": n_self,
        "self_correct_pct": C.pct(n_self, len(departures)),
        "n_generic_correct": n_generic,
        "generic_correct_pct": C.pct(n_generic, len(departures)),
        "net_cells_gained": n_self - n_generic,
    }
RES["self_vs_own_generic"] = same_as_generic
RES["departure_payoff"] = payoff


# Is an answer about itself distinctive?
# self_vs_own_generic can't be read alone, since nothing says whether 75% is high. Holding
# each answer in a cell out in turn and comparing it against the majority of the rest scores
# a self-prediction and a cross-prediction by one rule, so the two are directly comparable.
# A cell needs four rows: one is held out, and a majority over fewer than three isn't the
# quantity self_vs_own_generic uses. Cells hold four or five rows, so the other-row count
# isn't four times the cell count and is emitted rather than derived. Matching rates mean the
# name in the prompt doesn't change the answer, and the self case isn't a special case.

held_out = {}
for p in MODELS:
    n_self = self_hit = n_other = other_hit = 0
    for (owner, _mz, _step), rows in all_cells.items():
        if owner != p or len(rows) < 4:
            continue
        if sum(1 for r in rows if r.target == p) != 1:
            continue
        for i, held in enumerate(rows):
            rest = C.modal_position(r.pred for j, r in enumerate(rows) if j != i)
            if held.target == p:
                n_self += 1
                self_hit += held.pred == rest
            else:
                n_other += 1
                other_hit += held.pred == rest
    held_out[p] = {
        "n_cells": n_self,
        "n_self_matching": self_hit,
        "self_matches_rest_pct": C.pct(self_hit, n_self),
        "n_other_rows": n_other,
        "n_other_matching": other_hit,
        "other_matches_rest_pct": C.pct(other_hit, n_other),
    }
RES["held_out_agreement"] = held_out


# Write

with open(os.path.join(OUT, "cross_structure.json"), "w") as f:
    json.dump(RES, f, indent=1)
