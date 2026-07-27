#!/usr/bin/env python3
"""
Headline digest
---

Pulls the headline numbers out of the results files into one artifact and prints a console
digest. It computes nothing new, so it reflects whatever the analysis scripts last produced.

Notes:
- run_all.py runs this last, once every other script has written its file
- a missing or partial results file degrades to absent keys rather than raising
- the roster is kept local, so this script never imports common and never loads the dataset

Output: analysis/results/HEADLINES.json
"""

import json
import os

R = os.path.join(os.path.dirname(__file__), "results")
# Deliberately doesn't import common: this is a pure aggregation over the results files and
# shouldn't load the dataset. Keep the roster local.
MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]


def load(name):
    """Parsed results file, or an empty dict if it doesn't exist."""
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


outcomes = load("outcomes.json")
stats = load("stats.json")
traces = load("traces.json")
prior = load("prior.json")
expl = load("exploration_strategy.json")
xs = load("cross_structure.json")
mz = load("maze_structure.json")
ps = load("per_step.json")
H = {
    "metadata": {
        **outcomes.get("metadata", {}),
        "experiment": "HEADLINES",
        "produced_by": "analysis/summary.py",
    }
}


# 1) privileged access in aggregate
H["self_vs_best_other"] = {
    t: {
        "gap": d["overall"]["gap"],
        "ci": [d["overall"]["ci_lo"], d["overall"]["ci_hi"]],
        "p": d["overall"]["p_value_cluster_perm"],
    }
    for t, d in stats.get("self_vs_best_other_paired", {}).items()
}


# 2) where the self-advantage lives (atypical vs default decision points, first-unvisited taxonomy)
_atyp = stats.get("self_advantage_by_move_type", {}).get("atypical", {})
H["self_advantage_atypical"] = {
    t: {
        "n": d["n"],
        "self_acc": d["self_acc"],
        "best_other": d["best_other"]["model"],
        "gap_vs_best_other": d["best_other"]["gap_vs_best_other"],
        "p_value_cluster_perm": d["best_other"]["p_value_cluster_perm"],
        "holm_p": _atyp.get("holm_adjusted_best_other", {}).get(t),
        "gap_vs_mean_other": d["mean_other"]["gap_vs_mean_other"],
    }
    for t, d in _atyp.items()
    if t != "holm_adjusted_best_other"
}


# 3) oracle: items only self gets right
H["only_self_correct_pct"] = {
    t: d["items_only_self_correct_pct"] for t, d in xs.get("oracle_ceiling", {}).items()
}


# 4) mechanism: rule-likeness -> predictability (model side and maze side)
H["rule_likeness_vs_predictability"] = {
    "pearson": expl.get("regularity_vs_predictability", {}).get(
        "pearson_default_rate_vs_predictability"
    ),
    "perm_p": expl.get("regularity_vs_predictability", {}).get(
        "perm_p_default_rate_vs_predictability"
    ),
}
H["maze_branch_steps_vs_predictability_intersection"] = mz.get(
    "correlations_intersection19", {}
).get("mean_branch_steps")


# 5) two reasoning architectures (chronology of the truth when wrong)
chrono = {}
for t, d in traces.get("self_traces", {}).items():
    c = d.get("chronology_when_wrong", {})
    tot = sum(c.values()) or 1
    chrono[t] = {
        "truth_never_appears_pct": round(100.0 * c.get("truth_never_appears", 0) / tot, 1),
        "n_wrong": sum(c.values()),
    }
H["chronology_when_wrong"] = chrono


# 6) the prior, and reasoning's relation to it
H["nr_vs_reasoning_cross_model_agreement"] = {
    "nr": prior.get("cross_model_agreement", {}).get("nr_pairwise_agreement_pct"),
    "reasoning": prior.get("cross_model_agreement", {}).get("reasoning_pairwise_agreement_pct"),
}
H["reasoning_vs_nr_gap"] = {
    m: d["gap"] for m, d in outcomes.get("reasoning_vs_nr_self", {}).items()
}


# 7) self looks like the consensus, not the truth
H["self_matches_truth_vs_consensus"] = {
    t: {"truth": d["self_matches_truth_pct"], "consensus": d["self_matches_consensus_pct"]}
    for t, d in xs.get("per_target_structure", {}).items()
}


# 8) noise floor
H["validation_self_consistency"] = {
    m: d["frac_all_runs_agree"] for m, d in stats.get("validation_self_consistency", {}).items()
}


# 9) developer affinity (only opus<->sonnet is same-developer; underpowered at n=2)
H["developer_affinity"] = {
    "same_developer_pairs": xs.get("developer_affinity", {}).get("same_developer_pairs"),
    "mean_residual_same_developer": xs.get("developer_affinity", {}).get(
        "mean_residual_same_developer"
    ),
    "mean_residual_different_developer": xs.get("developer_affinity", {}).get(
        "mean_residual_different_developer"
    ),
}


# 10) self vs other-prediction dissociation, with the joint-fit predictor effects as the
#     difficulty-controlled simulator-skill measure
H["self_vs_other_prediction_dissociation"] = xs.get("self_vs_other_prediction_dissociation", {})
H["predictor_effects"] = xs.get("predictor_target_specialization", {}).get("predictor_effect")


# 11) predictability decay shape per step (where each target cliffs)
H["target_predictability_per_step"] = outcomes.get("target_predictability_per_step", {})


# 12) why the mid-horizon window: gap tracks prior collapse, not branch density
H["midhorizon_explanation"] = {
    t: {
        "corr_gap_vs_branch_rate": v.get("corr_gap_vs_branch_rate"),
        "corr_gap_vs_prior_nr": v.get("corr_gap_vs_prior_nr"),
    }
    for t, v in expl.get("self_advantage_vs_branch_rate", {}).items()
}


# 13) per-step cross-model convergence (NR prior vs reasoned), and late-horizon
#     below-baseline caveat
H["cross_model_agreement_by_step"] = {
    "nr": prior.get("cross_model_agreement", {}).get("by_step_nr"),
    "reasoning": prior.get("cross_model_agreement", {}).get("by_step_reasoning"),
}
# 14) error propagation, predictability horizon, hedging calibration, self-projection,
#     maze-difficulty-is-shared
H["error_propagation"] = ps.get("error_propagation", {})
H["predictability_horizon"] = {
    t: d.get("first_step_below_50pct") for t, d in ps.get("predictability_horizon", {}).items()
}
H["hedging_calibration"] = traces.get("hedging_calibration", {})
H["self_projection"] = {
    "pearson_similarity_vs_cross_acc": xs.get("self_projection", {}).get(
        "pearson_similarity_vs_cross_acc"
    ),
    "pearson_similarity_vs_residual": xs.get("self_projection", {}).get(
        "pearson_similarity_vs_residual"
    ),
}
H["self_vs_cross_predictability_per_maze"] = mz.get(
    "self_vs_cross_predictability_per_maze", {}
).get("pearson")
with open(os.path.join(R, "HEADLINES.json"), "w") as f:
    json.dump(H, f, indent=1)


if __name__ == "__main__":
    print("HEADLINE FINDINGS DIGEST")
    print("=" * 70)
    print("\n1) Self vs best-other in aggregate (paired):")
    for t in MODELS:
        d = H["self_vs_best_other"].get(t)
        if d:
            print(
                f"   {t:7} gap {d['gap']:+5.1f}  "
                f"CI[{d['ci'][0]:+.1f},{d['ci'][1]:+.1f}]  p={d['p']}"
            )

    print("\n2) Atypical decision points (first-unvisited taxonomy), self vs best-other:")
    for t in MODELS:
        d = H["self_advantage_atypical"][t]
        print(
            f"   {t:7} n={d['n']:3}  self {d['self_acc']:5}  vs best({d['best_other']}) "
            f"gap {d['gap_vs_best_other']:+.1f}  p={d['p_value_cluster_perm']}  "
            f"holm={d['holm_p']}"
        )

    print("\n3) Oracle: % of items only self predicts correctly:")
    print("   " + "  ".join(f"{t} {H['only_self_correct_pct'].get(t)}%" for t in MODELS))

    print("\n4) Rule-likeness vs predictability:")
    fm = H["rule_likeness_vs_predictability"]
    bm = H["maze_branch_steps_vs_predictability_intersection"] or {}
    print(
        f"   model side: corr(default rate, predictability) r={fm['pearson']} "
        f"(perm p={fm['perm_p']})"
    )
    print(
        f"   maze side : corr(branch decisions, predictability)  r={bm.get('pearson')} "
        f"(perm p={bm.get('perm_p')})"
    )

    print("\n5) Truth never appears in wrong traces (% of wrong traces):")
    print(
        "   "
        + "  ".join(
            f"{t} {H['chronology_when_wrong'].get(t, {}).get('truth_never_appears_pct')}%"
            for t in MODELS
        )
    )

    print("\n6) The prior and reasoning:")
    ag = H["nr_vs_reasoning_cross_model_agreement"]
    print(
        f"   cross-model agreement: NR priors {ag['nr']}%  vs  reasoned {ag['reasoning']}%"
    )
    print(
        "   reasoning - no-reasoning self gap: "
        + "  ".join(f"{m} {H['reasoning_vs_nr_gap'].get(m):+}" for m in MODELS)
    )

    print("\n7) Self matches truth vs consensus-of-others:")
    for t in MODELS:
        d = H["self_matches_truth_vs_consensus"].get(t)
        if d:
            print(f"   {t:7} truth {d['truth']}%   consensus {d['consensus']}%")

    print("\n8) Temp-0 noise floor (fraction of validation cells fully agreeing):")
    print("   " + "  ".join(f"{m} {H['validation_self_consistency'].get(m)}" for m in MODELS))

    da = H["developer_affinity"]
    print("\n9) Developer affinity (only opus<->sonnet is same-developer, and asymmetric):")
    print(
        f"   same-developer pairs {da['same_developer_pairs']}  "
        f"(mean {da['mean_residual_same_developer']})"
    )

    print("\n10) Self vs other-prediction skill (raw, and joint-fit predictor effects):")
    for m in MODELS:
        d = H["self_vs_other_prediction_dissociation"].get(m, {})
        print(
            f"   {m:7} self {d.get('self_acc')}  "
            f"raw-other {d.get('mean_acc_predicting_others')}"
        )
    pe = H.get("predictor_effects") or {}
    print(f"   joint-fit predictor effects: {pe}")

    print("\n11) Predictability decay by step (mean over predictors):")
    for t in MODELS:
        print(f"   {t:7} {H['target_predictability_per_step'].get(t)}")

    print("\n12) Mid-horizon gap vs branch rate and prior (opus):")
    mh = H["midhorizon_explanation"].get("opus", {})
    print(
        f"   opus corr(gap, branch_rate)={mh.get('corr_gap_vs_branch_rate')}  "
        f"corr(gap, prior_acc)={mh.get('corr_gap_vs_prior_nr')}"
    )

    print("\n13) Per-step cross-model convergence (NR prior vs reasoned):")
    print(f"   NR : {H['cross_model_agreement_by_step']['nr']}")
    print(f"   R  : {H['cross_model_agreement_by_step']['reasoning']}")

    print("\n14) Error propagation P(next correct | this correct vs wrong):")
    for m in MODELS:
        d = H["error_propagation"].get(m, {})
        print(
            f"   {m:7} given-correct {d.get('p_correct_next_given_correct')}%  "
            f"given-wrong {d.get('p_correct_next_given_wrong')}%"
        )

    print("\n15) Accuracy when hedging vs not:")
    for m in MODELS:
        d = H["hedging_calibration"].get(m, {})
        print(
            f"   {m:7} hedging {d.get('acc_when_hedging')}%  not {d.get('acc_when_not_hedging')}%"
        )

    print(
        "\n16) Self-projection sim-vs-residual and per-maze self-vs-cross: r =",
        H["self_projection"]["pearson_similarity_vs_residual"],
        "| r =",
        H["self_vs_cross_predictability_per_maze"],
    )

    print("\n-> wrote results/HEADLINES.json")
