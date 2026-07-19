#!/usr/bin/env python3
"""
Run the full analysis pipeline
==============================

Executes every analysis script in order; each imports common.py (the single source of the
scoring contract) and writes its own results/<name>.json. Run from anywhere:

    python3 analysis/run_all.py

Each script also prints a short console summary. A non-zero exit from any script stops the
run so a failure is never silently skipped.
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "outcomes.py",  # accuracy matrix, self-vs-other, difficulty strata, reasoning gradient
    "prior.py",  # the no-reasoning prior: concentration, individuation, rescue/trap
    "stats.py",  # paired tests, CIs, noise floor, baselines, prior-alignment split
    "error_geometry.py",  # distance / off-by-step / reachability / distribution / corner
    "exploration_strategy.py",  # forced-vs-branch, regularity->predictability, determinism
    "maze_structure.py",  # maze-side: what makes a maze predictable, net of navigator
    "traces.py",  # trace features raw+normalized, chronology, path-simulation, length
    "cross_structure.py",  # convergence, ensembles, oracle, specialization, dissociation
    "per_step.py",  # horizon-resolved: consistency/forced-branch/consensus by step, propagation
]
# run last: reads every results/*.json and writes a consolidated headline digest
FINAL = "summary.py"


def main():
    # sanity-check the foundation first
    print("=" * 70 + "\ncommon.py (foundation)\n" + "=" * 70)
    if subprocess.run([sys.executable, os.path.join(HERE, "common.py")]).returncode != 0:
        sys.exit("FAILED: common.py")
    for s in SCRIPTS:
        print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)
        if subprocess.run([sys.executable, os.path.join(HERE, s)]).returncode != 0:
            sys.exit(f"FAILED: {s}")
    print("\n" + "=" * 70 + f"\n{FINAL}\n" + "=" * 70)
    if subprocess.run([sys.executable, os.path.join(HERE, FINAL)]).returncode != 0:
        sys.exit(f"FAILED: {FINAL}")
    print("\n" + "=" * 70)
    print("All analyses complete. Results written to analysis/results/.")


if __name__ == "__main__":
    main()
