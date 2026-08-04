#!/usr/bin/env python3
"""
Run-substitution validation
---

Recomputes the whole pipeline three times, once per self-prediction run, and reports the
headline quantities side by side. Run it from anywhere with python3
analysis/validate_substitution.py.

Method:
- each variant runs run_all.py with RUN_PREF set to 0, 1 or 2, which changes which prediction
  run supplies every scored position, across all twenty predictor-target cells at once
- a cell with no parsed record for the preferred run falls back to run 0, so every variant
  scores the same cells and the comparison is like-for-like
- variant trees are written to a temporary directory; only the comparison file is kept
- SEED, N_PERM and B are untouched, so any movement comes from the substituted positions

Coverage:
- alternate runs exist for roughly a fifth of cells, the validation subsample
- within those the models mostly repeat themselves, so the share of scored positions that
  actually change is small; the emitted coverage block reports it exactly, and any claim
  about surviving substitution should be read against that number

Output: analysis/results/validation_substitution.json
"""

import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "..", "data")
MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]
VARIANTS = (0, 1, 2)
OUTPUT = os.path.join(RESULTS, "validation_substitution.json")


def substitution_coverage():
    """Cells carrying each alternate run, and how many of them change the scored position."""
    counts = collections.Counter()
    for m in MODELS:
        for path in (
            os.path.join(DATA, "self_prediction", f"{m}_self_reasoning.json"),
            os.path.join(DATA, "cross_prediction", f"{m}_xpred_reasoning.json"),
        ):
            for cell in json.load(open(path))["predictions"].values():
                for steps in cell.values():
                    for recs in steps.values():
                        by = {
                            r.get("run_idx"): r.get("parsed_position")
                            for r in recs
                            if r.get("parsed_position") is not None
                        }
                        if 0 not in by:
                            continue
                        counts["n_cells"] += 1
                        for run in (1, 2):
                            if run in by:
                                counts[f"n_with_run_{run}"] += 1
                                if by[run] != by[0]:
                                    counts[f"n_changed_run_{run}"] += 1
    out = {"n_cells": counts["n_cells"]}
    for run in (1, 2):
        have, changed = counts[f"n_with_run_{run}"], counts[f"n_changed_run_{run}"]
        out[f"run_{run}"] = {
            "n_cells_with_run": have,
            "pct_cells_with_run": round(100.0 * have / counts["n_cells"], 2),
            "n_positions_changed": changed,
            "pct_positions_changed": round(100.0 * changed / counts["n_cells"], 2),
        }
    return out


def run_variant(pref, dest):
    """Run the pipeline with RUN_PREF=pref and copy its results tree to dest."""
    env = dict(os.environ, RUN_PREF=str(pref))
    subprocess.run(
        [sys.executable, os.path.join(HERE, "run_all.py")],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    os.makedirs(dest, exist_ok=True)
    for name in sorted(os.listdir(RESULTS)):
        if name.endswith(".json") and name != os.path.basename(OUTPUT):
            shutil.copy(os.path.join(RESULTS, name), os.path.join(dest, name))


def load(tree, name):
    """One results file from a variant tree."""
    return json.load(open(os.path.join(tree, name)))


def claims_for(tree):
    """The quantities tracked across variants, pulled from one variant's results tree."""
    stats = load(tree, "stats.json")
    outcomes = load(tree, "outcomes.json")
    expl = load(tree, "exploration_strategy.json")
    xs = load(tree, "cross_structure.json")

    atypical = stats["self_advantage_by_move_type"]["atypical"]
    opus_atypical = atypical["opus"]
    unique = stats["unique_information"]["opus"]
    strongest = unique["strongest_other"]
    reg = expl["rule_likeness_vs_predictability_cross"]
    tracking = xs["cross_acc_vs_target_self_acc"]

    out = {
        "opus_atypical_self_acc": opus_atypical["self_acc"],
        "opus_atypical_n_cells": opus_atypical["n"],
        "opus_atypical_holm_p_best_other": atypical["holm_adjusted_best_other"]["opus"],
        "opus_unique_correct": unique["unique_correct"]["opus"],
        "opus_unique_strongest_other": strongest,
        "opus_unique_margin": unique["count_diff_vs_strongest"],
        "opus_unique_max_p": unique["max_p"],
        "default_rate_vs_cross_acc_pearson": reg["pearson"],
        "default_rate_vs_cross_acc_perm_p": reg["perm_p"],
        "self_acc_vs_cross_acc_pearson": tracking["pearson"],
        "self_acc_vs_cross_acc_perm_p": tracking["perm_p"],
    }
    for p, row in opus_atypical["per_predictor"].items():
        out[f"opus_atypical_gap_vs_{p}"] = row["gap"]
        out[f"opus_atypical_p_vs_{p}"] = row["p_value_cluster_perm"]
    for predictor, row in outcomes["matrix_native"].items():
        for target, acc in row.items():
            out[f"matrix_native_{predictor}_to_{target}"] = acc
    return out


def main():
    """Run every variant, then write the side-by-side comparison."""
    workdir = tempfile.mkdtemp(prefix="run_substitution_")
    try:
        per_variant = {}
        for pref in VARIANTS:
            print(f"running variant RUN_PREF={pref}")
            tree = os.path.join(workdir, f"run_{pref}")
            run_variant(pref, tree)
            per_variant[pref] = claims_for(tree)
    finally:
        # leave analysis/results holding the committed variant whatever happened above
        run_variant(0, os.path.join(workdir, "restore"))
        shutil.rmtree(workdir, ignore_errors=True)

    names = list(per_variant[0])
    claims = {}
    for name in names:
        values = {f"run_{pref}": per_variant[pref][name] for pref in VARIANTS}
        numeric = [v for v in values.values() if isinstance(v, (int, float))]
        claims[name] = {
            **values,
            "identical": len(set(map(str, values.values()))) == 1,
            "max_abs_change": (
                round(max(abs(v - numeric[0]) for v in numeric), 4) if numeric else None
            ),
        }
    unstable = sorted(k for k, v in claims.items() if not v["identical"])
    payload = {
        "metadata": {
            "experiment": "validation_substitution",
            "produced_by": "analysis/validate_substitution.py",
            "variants": list(VARIANTS),
            "n_claims": len(claims),
            "n_claims_identical": len(claims) - len(unstable),
        },
        "coverage": substitution_coverage(),
        "claims_that_moved": unstable,
        "claims": claims,
    }
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n{len(claims) - len(unstable)} of {len(claims)} tracked values identical")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
