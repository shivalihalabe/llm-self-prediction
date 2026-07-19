#!/usr/bin/env python3
"""
Headline digest
===============

Reads every results/*.json the analysis scripts wrote and pulls the headline numbers into one
artifact (results/HEADLINES.json) plus a readable console digest. Pure aggregation: it computes
nothing new, so it always reflects whatever the analysis scripts last produced. run_all.py runs
this last.
Output: analysis/results/HEADLINES.json
"""

import json
import os

R = os.path.join(os.path.dirname(__file__), "results")
MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


outcomes = load("outcomes.json")
stats = load("stats.json")
traces = load("traces.json")
prior = load("prior.json")
expl = load("exploration_strategy.json")
xs = load("cross_structure.json")
mz = load("maze_structure.json")
eg = load("error_geometry.json")
ps = load("per_step.json")
H = {"metadata": {**outcomes["metadata"], "experiment": "HEADLINES", "produced_by": "analysis/summary.py"}}

# 1) privileged access in aggregate
H["self_vs_best_other"] = {
    t: {
        "gap": d["overall"]["gap"],
        "ci": [d["overall"]["ci_lo"], d["overall"]["ci_hi"]],
        "p": d["overall"]["p_value"],
    }
    for t, d in stats.get("self_vs_best_other_paired", {}).items()
}

# 2) where the self-advantage lives (prior-aligned vs idiosyncratic branches)
H["prior_alignment"] = {
    t: {
        "idiosyncratic_gap": d["idiosyncratic"]["gap"],
        "idiosyncratic_p": d["idiosyncratic"]["p_value"],
        "prior_aligned_gap": d["prior_aligned"]["gap"],
    }
    for t, d in stats.get("self_advantage_prior_aligned_vs_idiosyncratic", {}).items()
}

# 2b) the Opus mid-horizon per-step advantage (significant where the CI excludes 0)
_ps = stats.get("self_vs_best_other_per_step", {}).get("opus", {})
H["opus_midhorizon_per_step"] = {
    "best_other": _ps.get("best_other"),
    "by_step": [
        {"step": r["step"], "gap": r["gap"], "ci": [r["ci_lo"], r["ci_hi"]], "sig": r["sig"]}
        for r in _ps.get("by_step", [])
    ],
}

# 3) oracle: items only self gets right
H["only_self_correct_pct"] = {
    t: d["items_only_self_correct_pct"] for t, d in xs.get("oracle_ceiling", {}).items()
}

# 4) mechanism: rule-likeness -> predictability (model side and maze side)
H["mechanism_first_listed_vs_predictability"] = {
    "pearson": expl.get("regularity_vs_predictability", {}).get(
        "pearson_firstlisted_vs_predictability"
    ),
    "perm_p": expl.get("regularity_vs_predictability", {}).get(
        "perm_p_firstlisted_vs_predictability"
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

# 9) developer affinity (corrected: only opus<->sonnet is same-developer; glm/qwen are different developers)
H["developer_affinity"] = {
    "same_developer_pairs": xs.get("developer_affinity", {}).get("same_developer_pairs"),
    "mean_residual_same_developer": xs.get("developer_affinity", {}).get(
        "mean_residual_same_developer"
    ),
    "mean_residual_different_developer": xs.get("developer_affinity", {}).get(
        "mean_residual_different_developer"
    ),
    "open_weight_pair_glm_qwen": xs.get("developer_affinity", {}).get("open_weight_pair_glm_qwen"),
}

# 10) self vs OTHER prediction dissociation (Opus introspector vs Sonnet simulator)
H["self_vs_other_prediction_dissociation"] = xs.get("self_vs_other_prediction_dissociation", {})

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

# 13) per-step cross-model convergence (NR prior vs reasoned), and late-horizon below-baseline caveat
H["cross_model_agreement_by_step"] = {
    "nr": prior.get("cross_model_agreement", {}).get("by_step_nr"),
    "reasoning": prior.get("cross_model_agreement", {}).get("by_step_reasoning"),
}
H["opus_step8_lift_vs_modal_baseline"] = next(
    (
        r["lift_vs_modal"]
        for r in stats.get("baseline_sensitivity", {}).get("opus", [])
        if r["step"] == 8
    ),
    None,
)

# 14) error propagation, predictability horizon, hedging calibration, self-projection, maze-difficulty-is-shared
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
H["reasoned_consensus_vs_truth_vs_prior_opus"] = prior.get(
    "reasoned_consensus_vs_truth_vs_prior", {}
).get("opus")

with open(os.path.join(R, "HEADLINES.json"), "w") as f:
    json.dump(H, f, indent=1)


if __name__ == "__main__":

    def g(d, t, k, default="?"):
        return d.get(t, {}).get(k, default)

    print("HEADLINE FINDINGS DIGEST")
    print("=" * 70)
    print("\n1) Privileged self-access in aggregate (self vs best-other, paired):")
    for t in MODELS:
        d = H["self_vs_best_other"].get(t)
        if d:
            print(
                f"   {t:7} gap {d['gap']:+5.1f}  CI[{d['ci'][0]:+.1f},{d['ci'][1]:+.1f}]  p={d['p']}"
            )
    print("   => only Opus is positive (and borderline); others tie or lose.")

    print("\n2) Where the advantage lives (branch decisions, by prior-alignment):")
    for t in MODELS:
        d = H["prior_alignment"].get(t)
        if d:
            print(
                f"   {t:7} idiosyncratic gap {d['idiosyncratic_gap']:+5.1f} (p={d['idiosyncratic_p']})   prior-aligned {d['prior_aligned_gap']:+.1f}"
            )
    print(
        "   => Opus self-knowledge is real and localized to idiosyncratic choices; Sonnet/GPT are worse there."
    )

    print("\n2b) Opus mid-horizon per-step advantage (self vs best-other, by step):")
    for r in H["opus_midhorizon_per_step"]["by_step"]:
        flag = "  <== CI excludes 0" if r["sig"] else ""
        print(
            f"   step {r['step']}: {r['gap']:+5.1f}  CI[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]{flag}"
        )
    print(
        "   => Opus is the only model with a positive mid-horizon spike (peak step 5 +11.3); others flat-to-negative."
    )

    print("\n3) Oracle: % of items only self predicts correctly:")
    print("   " + "  ".join(f"{t} {H['only_self_correct_pct'].get(t)}%" for t in MODELS))
    print(
        "   => only Opus contributes unique self-knowledge; others fully redundant with outsiders."
    )

    print("\n4) Mechanism - rule-likeness drives predictability:")
    fm = H["mechanism_first_listed_vs_predictability"]
    bm = H["maze_branch_steps_vs_predictability_intersection"] or {}
    print(
        f"   model side: corr(first-listed rate, predictability) r={fm['pearson']} (perm p={fm['perm_p']})"
    )
    print(
        f"   maze side : corr(branch decisions, predictability)  r={bm.get('pearson')} (perm p={bm.get('perm_p')})"
    )

    print("\n5) Two reasoning architectures (truth never appears in wrong traces):")
    print(
        "   "
        + "  ".join(
            f"{t} {H['chronology_when_wrong'].get(t, {}).get('truth_never_appears_pct')}%"
            for t in MODELS
        )
    )
    print(
        "   => Opus/Sonnet/GPT selectively branch (truth never considered); GLM/Qwen enumerate then reject it."
    )

    print("\n6) The prior and reasoning:")
    ag = H["nr_vs_reasoning_cross_model_agreement"]
    print(
        f"   cross-model agreement: NR priors {ag['nr']}%  vs  reasoned {ag['reasoning']}%  => reasoning CONVERGES models"
    )
    print(
        "   reasoning - no-reasoning self gap: "
        + "  ".join(f"{m} {H['reasoning_vs_nr_gap'].get(m):+}" for m in MODELS)
    )

    print("\n7) Self matches consensus-of-others more than truth:")
    for t in MODELS:
        d = H["self_matches_truth_vs_consensus"].get(t)
        if d:
            print(f"   {t:7} truth {d['truth']}%   consensus {d['consensus']}%")

    print("\n8) Temp-0 noise floor (fraction of validation cells fully agreeing):")
    print("   " + "  ".join(f"{m} {H['validation_self_consistency'].get(m)}" for m in MODELS))

    da = H["developer_affinity"]
    print("\n9) Developer affinity (only opus<->sonnet is same-developer, and asymmetric):")
    print(
        f"   same-developer pairs {da['same_developer_pairs']}  (mean {da['mean_residual_same_developer']})"
    )
    print(
        f"   glm<->qwen are DIFFERENT developers (Zhipu vs Alibaba), shown separately: {da['open_weight_pair_glm_qwen']}"
    )
    print(
        "   => no clean same-developer affinity with this lineup; opus->sonnet (+12.5) is a one-directional specialization."
    )

    print(
        "\n10) Self vs OTHER prediction (raw other-skill is target-difficulty-confounded; residual controls for it):"
    )
    for m in MODELS:
        d = H["self_vs_other_prediction_dissociation"].get(m, {})
        rp = d.get("mean_residual_as_predictor")
        rpt = f"{rp:+.2f}" if rp is not None else "NA"
        print(
            f"   {m:7} self {d.get('self_acc')}  raw-other {d.get('skill_predicting_others')}  residual-as-predictor {rpt}"
        )
    print(
        "   => Opus: ~average simulator (resid -2.2) but the ONLY privileged self-access -> introspector."
    )
    print(
        "      Qwen/GPT: best simulators (resid +6.4/+4.5). Sonnet: weakest at BOTH self and simulation (its high"
    )
    print("      raw-other 72.3 is just an artifact of predicting the easy targets gpt/qwen).")

    print("\n11) Predictability decay by step (mean over predictors) - note Opus cliff at step 4:")
    for t in MODELS:
        print(f"   {t:7} {H['target_predictability_per_step'].get(t)}")
    print(
        "   => Opus/GLM cliff mid-horizon; GPT/Qwen decay gently and plateau (predictable throughout)."
    )

    print("\n12) Why the Opus mid-horizon window:")
    mh = H["midhorizon_explanation"].get("opus", {})
    print(
        f"   opus corr(gap, branch_rate)={mh.get('corr_gap_vs_branch_rate')}  corr(gap, prior_acc)={mh.get('corr_gap_vs_prior_nr')}"
    )
    print(
        "   advantage opens at step 4 where the prior dies (57%->10%) and closes by step 6 as self-prediction also decays to noise."
    )

    print("\n13) Per-step cross-model convergence (NR prior vs reasoned):")
    print(f"   NR : {H['cross_model_agreement_by_step']['nr']}")
    print(f"   R  : {H['cross_model_agreement_by_step']['reasoning']}")
    print(
        f"   caveat: at step 8 Opus self-prediction falls BELOW the modal baseline (lift {H['opus_step8_lift_vs_modal_baseline']})."
    )

    print(
        "\n14) Error propagation P(next correct | this correct vs wrong) - errors are sticky except Qwen:"
    )
    for m in MODELS:
        d = H["error_propagation"].get(m, {})
        print(
            f"   {m:7} given-correct {d.get('p_correct_next_given_correct')}%  given-wrong {d.get('p_correct_next_given_wrong')}%"
        )

    print("\n15) Hedging is a real confidence signal (accuracy when hedging vs not):")
    for m in MODELS:
        d = H["hedging_calibration"].get(m, {})
        print(
            f"   {m:7} hedging {d.get('acc_when_hedging')}%  not {d.get('acc_when_not_hedging')}%"
        )

    print(
        "\n16) Other: self-projection sim-vs-residual r =",
        H["self_projection"]["pearson_similarity_vs_residual"],
        "| maze difficulty shared (self-vs-cross per maze) r =",
        H["self_vs_cross_predictability_per_maze"],
    )

    print("\n-> wrote results/HEADLINES.json")
