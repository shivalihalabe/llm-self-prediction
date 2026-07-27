#!/usr/bin/env python3
"""
Cross-prediction structure
==========================
Relationships between predictors.

For each target: how tightly all five predictors (incl. self) converge in accuracy, whether
they agree on the same coordinate (not just both-right), whether a majority-vote ensemble
beats the best single predictor, whether wrong predictors share the same wrong answer, and
whether self's prediction looks more like the truth or like the consensus of others. Plus two
aggregate relationships: the A->B vs B->A asymmetry tracking target predictability, and
cross-accuracy tracking the target's own self-accuracy.

Uses the predicted-coordinate dicts (common.SELF_POS / CROSS_POS).

Output: analysis/results/cross_structure.json
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


# ============================================================
# PER-TARGET STRUCTURE
# ============================================================

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

    # inter-predictor position agreement (all pairs) + agreement among the jointly-wrong
    pair_agree = pair_total = wrong_pair_agree = wrong_pair_total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            same = wide[a].eq(wide[b])
            pair_agree += int(same.sum())
            pair_total += n
            both_wrong = ~correct[a] & ~correct[b]
            wrong_pair_agree += int((same & both_wrong).sum())
            wrong_pair_total += int(both_wrong.sum())

    # ensemble majority vote (ties break by predictor order, as in Counter insertion)
    mv_others = wide[others].apply(lambda r: collections.Counter(r).most_common(1)[0][0], axis=1)
    mv_all = wide[names].apply(lambda r: collections.Counter(r).most_common(1)[0][0], axis=1)

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
        "best_single_acc": max(accs) if accs else None,
        "ensemble_others_acc": C.pct(mv_others.eq(wide["truth"]).mean()) if n else None,
        "ensemble_all_acc": C.pct(mv_all.eq(wide["truth"]).mean()) if n else None,
        "self_matches_truth_pct": C.pct(correct["self"].mean()) if n else None,
        "self_matches_consensus_pct": C.pct(wide["self"].eq(mv_others).mean()) if n else None,
    }
RES["per_target_structure"] = struct


# ============================================================
# ASYMMETRY TRACKS PREDICTABILITY
# ============================================================

# For each unordered pair, (A->B minus B->A) should track (predictability_B minus predictability_A).
predictability = {}
for t in MODELS:
    vals = [C.acc(C.SELF[t])[0]] + [
        C.acc(C.CROSS[(p, t)])[0] for p in MODELS if p != t and (p, t) in C.CROSS
    ]
    predictability[t] = st.mean([v for v in vals if v is not None])

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
        acc_gaps.append(ab - ba)
        pred_gaps.append(predictability[b] - predictability[a])
        pairs_dump.append(
            {
                "pair": f"{a}|{b}",
                "a_to_b": round(ab, 1),
                "b_to_a": round(ba, 1),
                "acc_gap": round(ab - ba, 1),
                "predictability_gap_b_minus_a": round(predictability[b] - predictability[a], 1),
            }
        )
RES["asymmetry_vs_predictability"] = {
    "pearson": pearson(pred_gaps, acc_gaps),
    "perm_p": C.perm_corr_p(pred_gaps, acc_gaps),
    "n_pairs": len(pred_gaps),
    "note": "positive => the direction pointing at the more predictable target is the higher one",
    "pairs": pairs_dump,
}


# ============================================================
# CROSS-ACCURACY TRACKS TARGET SELF-ACCURACY
# ============================================================

xs, ys = [], []
for p in MODELS:
    for t in MODELS:
        if t == p or (p, t) not in C.CROSS:
            continue
        cr = C.acc(C.CROSS[(p, t)])[0]
        sf = C.acc(C.SELF[t])[0]
        if cr is not None and sf is not None:
            xs.append(sf)
            ys.append(cr)
RES["cross_acc_vs_target_self_acc"] = {
    "pearson": pearson(xs, ys),
    "perm_p": C.perm_corr_p(xs, ys),
    "note": "high => predicting a model is governed by that model's own predictability (simulation, not self-knowledge)",
    "n_cells": len(xs),
}


# ============================================================
# ORACLE CEILING + SELF'S UNIQUE CONTRIBUTION
# ============================================================

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


# ============================================================
# PREDICTOR X TARGET SPECIALIZATION
# ============================================================

# Residual of each cross cell after an additive predictor-skill + target-predictability model.
cells = {
    (p, t): C.acc(C.CROSS[(p, t)])[0]
    for p in MODELS
    for t in MODELS
    if t != p and (p, t) in C.CROSS
}
grand = st.mean(cells.values())
skill = {p: st.mean([cells[(p, t)] for t in MODELS if (p, t) in cells]) for p in MODELS}
tpred = {t: st.mean([cells[(p, t)] for p in MODELS if (p, t) in cells]) for t in MODELS}
resid = {f"{p}->{t}": round(v - (skill[p] + tpred[t] - grand), 1) for (p, t), v in cells.items()}
RES["predictor_target_specialization"] = {
    "grand_mean": round(grand, 1),
    "predictor_skill": {p: round(v, 1) for p, v in skill.items()},
    "target_predictability": {t: round(v, 1) for t, v in tpred.items()},
    "residuals": resid,
    "note": "residual = cross_acc - (predictor_skill + target_predictability - grand_mean); + => affinity beyond main effects",
}


# ============================================================
# DEVELOPER AFFINITY (same developer vs different)
# ============================================================

# Correct developers: GLM is Zhipu, Qwen is Alibaba -- DIFFERENT companies. The only same-developer
# pair in the lineup is opus<->sonnet (Anthropic), so a same-developer affinity test is underpowered
# and the opus<->sonnet relationship is asymmetric. The glm<->qwen pair (both open-weight, but
# different developers) is reported separately rather than miscounted as same-developer.
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
        round(st.mean([resid[k] for k in same_dev]), 2) if same_dev else None
    ),
    "mean_residual_different_developer": (
        round(st.mean([resid[k] for k in diff_dev]), 2) if diff_dev else None
    ),
    "open_weight_pair_glm_qwen": {
        k: resid[k] for k in resid if set(k.split("->")) == {"glm", "qwen"}
    },
    "note": "developers: anthropic={opus,sonnet}, openai={gpt}, zhipu={glm}, alibaba={qwen}. Only opus<->sonnet is same-developer (n=2, asymmetric: opus->sonnet +12.5 vs sonnet->opus -9.7) so this is underpowered. GLM and Qwen are DIFFERENT developers; their pair is listed separately, not as same-developer.",
}


# ============================================================
# SELF VS OTHER-PREDICTION DISSOCIATION
# ============================================================

# Is a model good at predicting itself, at predicting others, both, or neither? (Opus vs Sonnet differ sharply.)
dissociation = {}
for m in MODELS:
    self_acc = C.acc(C.SELF[m])[0]
    as_predictor = [C.acc(C.CROSS[(m, t)])[0] for t in MODELS if t != m and (m, t) in C.CROSS]
    as_predictor = [v for v in as_predictor if v is not None]
    as_target = [C.acc(C.CROSS[(p, m)])[0] for p in MODELS if p != m and (p, m) in C.CROSS]
    as_target = [v for v in as_target if v is not None]
    res_as_pred = [resid[f"{m}->{t}"] for t in MODELS if f"{m}->{t}" in resid]
    dissociation[m] = {
        "self_acc": round(self_acc, 1),
        "skill_predicting_others": round(st.mean(as_predictor), 1) if as_predictor else None,
        "predictability_by_others": round(st.mean(as_target), 1) if as_target else None,
        "self_minus_other_skill": (
            round(self_acc - st.mean(as_predictor), 1) if as_predictor else None
        ),
        "mean_residual_as_predictor": round(st.mean(res_as_pred), 2) if res_as_pred else None,
    }
RES["self_vs_other_prediction_dissociation"] = dissociation


# ============================================================
# IS CROSS-PREDICTION SELF-PROJECTION?
# ============================================================


# Does a predictor predict a target better when the target actually behaves like the predictor?
def traj_similarity(p, t):
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
        resids.append(resid[f"{p}->{t}"])
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
    "note": "positive residual correlation => predictors have genuine affinity for behaviorally-similar targets (self-projection) beyond general predictability",
}


# ============================================================
# PAIRWISE BEHAVIORAL SIMILARITY MATRIX
# ============================================================

# The trajectory-overlap matrix behind the self-projection correlation: traj_similarity for
# every ordered pair (pooled per-step position match over shared consistent mazes, excluding
# the shared start cell).
RES["behavioral_similarity_matrix"] = {
    a: {b: round(traj_similarity(a, b), 1) for b in MODELS if b != a} for a in MODELS
}


# ============================================================
# STEP AGREEMENT ON ALL 100 MAZES AND ORACLE COMPOSITION BY MOVE TYPE
# ============================================================

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


# ============================================================
# WRITE
# ============================================================

with open(os.path.join(OUT, "cross_structure.json"), "w") as f:
    json.dump(RES, f, indent=1)
