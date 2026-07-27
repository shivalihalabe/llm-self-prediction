#!/usr/bin/env python3
"""
Shared foundation for all maze self-prediction analyses
=======================================================

Locks the scoring contract in ONE place. Everything downstream reads the tidy record
table (`RECORDS`) and the derived views built here, so "correct", "predicted position",
and "which mazes" mean the same thing in every script.

Scoring contract
-----------------
- run_idx == 0 records only (runs 1, 2 are reserved for the noise-floor analysis).
- per-step EXACT position match against the target's run-0 navigation trajectory.
- unparsed records (parsed_position is None) are skipped.
- maze-set treatments: native (target's consistent set), 5-way intersection, pairwise.

Core objects
------------
RECORDS : tidy DataFrame, one row per scored run-0 prediction
          [kind, predictor, target, maze, step, pred, truth, correct]
SELF[m], SELF_NR[m], CROSS[(p, t)], PILOT[(m, fr)]  -> {(maze, step): correct}
SELF_POS / CROSS_POS / ...                          -> {(maze, step): [r, c]}
CONSISTENT[m], INTERSECTION, PAIRWISE[(a, b)], MAZE_DIFFICULTY, DIFFICULTY_STRATA
TRUTH[m][maze] -> trajectory; WALLS[maze] -> set of blocked frozensets

Run scripts from the repo root; data/ is resolved relative to this file.
"""

import os
import json
import warnings
import collections

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy as _scipy_entropy, permutation_test, ConstantInputWarning

ROWS = COLS = 5
MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "..", "data")  # analysis/ sits at the repo root next to data/


# ============================================================
# NAVIGATION GROUND TRUTH
# ============================================================


def _nav(model):
    return json.load(open(os.path.join(DATA, "navigation", f"{model}_navigation.json")))


TRUTH = {
    m: {mz: node["runs"][0]["trajectory"] for mz, node in _nav(m)["navigation"][m].items()}
    for m in MODELS
}
WALLS = {
    mz["id"]: {frozenset([tuple(a), tuple(b)]) for a, b in mz["walls"]}
    for mz in _nav(MODELS[0])["mazes"]
}


# ============================================================
# BUILD THE TIDY RECORD TABLE
# ============================================================


def _iter_scored(node, target, listed):
    """Yield (maze, step, pred_tuple, correct) for run-0 parsed records under a prediction node.

    `listed` distinguishes the two on-disk shapes: reasoning/no-reasoning store a list of
    runs per (maze, step); the pilot stores a single record.
    """
    truth = TRUTH[target]
    for mz, steps in node.items():
        if mz not in truth:
            continue
        for stp, payload in steps.items():
            si = int(stp.split("_")[1])
            if si >= len(truth[mz]):
                continue
            rec = next((r for r in payload if r.get("run_idx") == 0), None) if listed else payload
            if not rec or rec.get("parsed_position") is None:
                continue
            pred = tuple(rec["parsed_position"])
            yield mz, si, pred, pred == tuple(truth[mz][si])


def _load_predictions(path):
    return json.load(open(path))["predictions"]


def _records():
    rows = []
    for m in MODELS:
        for mode, kind in [("reasoning", "self"), ("noreasoning", "self_nr")]:
            node = _load_predictions(
                os.path.join(DATA, "self_prediction", f"{m}_self_{mode}.json")
            )[f"{m}_self_{mode}"]
            for mz, si, pred, ok in _iter_scored(node, m, listed=True):
                rows.append((kind, m, m, mz, si, pred, ok))
    for p in MODELS:
        xp = _load_predictions(os.path.join(DATA, "cross_prediction", f"{p}_xpred_reasoning.json"))
        for t in MODELS:
            cell = f"{p}_xpred_{t}_reasoning"
            if cell not in xp:
                continue
            for mz, si, pred, ok in _iter_scored(xp[cell], t, listed=True):
                rows.append(("cross", p, t, mz, si, pred, ok))
    pilot = _load_predictions(os.path.join(DATA, "self_framing_pilot.json"))
    for m in MODELS:
        for fr, node in pilot[m].items():
            for mz, si, pred, ok in _iter_scored(node, m, listed=False):
                rows.append((f"pilot:{fr}", m, m, mz, si, pred, ok))
    df = pd.DataFrame(
        rows, columns=["kind", "predictor", "target", "maze", "step", "pred", "correct"]
    )
    df["truth"] = [tuple(TRUTH[t][mz][s]) for t, mz, s in zip(df.target, df.maze, df.step)]
    return df


RECORDS = _records()


# ============================================================
# DICT VIEWS (stable API)
# ============================================================


def _as_dicts(frame):
    scored = {(mz, s): bool(c) for mz, s, c in zip(frame.maze, frame.step, frame.correct)}
    pos = {(mz, s): list(p) for mz, s, p in zip(frame.maze, frame.step, frame.pred)}
    return scored, pos


SELF, SELF_POS, SELF_NR, SELF_NR_POS = {}, {}, {}, {}
for _m in MODELS:
    SELF[_m], SELF_POS[_m] = _as_dicts(RECORDS[(RECORDS.kind == "self") & (RECORDS.target == _m)])
    SELF_NR[_m], SELF_NR_POS[_m] = _as_dicts(
        RECORDS[(RECORDS.kind == "self_nr") & (RECORDS.target == _m)]
    )

CROSS, CROSS_POS = {}, {}
for _row in (
    RECORDS[RECORDS.kind == "cross"][["predictor", "target"]]
    .drop_duplicates()
    .itertuples(index=False)
):
    _sub = RECORDS[
        (RECORDS.kind == "cross")
        & (RECORDS.predictor == _row.predictor)
        & (RECORDS.target == _row.target)
    ]
    CROSS[(_row.predictor, _row.target)], CROSS_POS[(_row.predictor, _row.target)] = _as_dicts(_sub)

PILOT, PILOT_POS = {}, {}
for _kind in RECORDS[RECORDS.kind.str.startswith("pilot:")].kind.unique():
    _fr = _kind.split(":", 1)[1]
    for _m in MODELS:
        _sub = RECORDS[(RECORDS.kind == _kind) & (RECORDS.target == _m)]
        if len(_sub):
            PILOT[(_m, _fr)], PILOT_POS[(_m, _fr)] = _as_dicts(_sub)


# ============================================================
# MAZE SETS
# ============================================================

CONSISTENT = {
    m: set(RECORDS[(RECORDS.kind == "self") & (RECORDS.target == m)].maze.unique()) for m in MODELS
}
INTERSECTION = set.intersection(*(CONSISTENT[m] for m in MODELS))
PAIRWISE = {
    tuple(sorted((a, b))): CONSISTENT[a] & CONSISTENT[b]
    for i, a in enumerate(MODELS)
    for b in MODELS[i + 1 :]
}

_ALL = set().union(*CONSISTENT.values())
MAZE_DIFFICULTY = {mz: sum(mz in CONSISTENT[m] for m in MODELS) for mz in _ALL}
DIFFICULTY_STRATA = {k: {mz for mz, c in MAZE_DIFFICULTY.items() if c == k} for k in range(1, 6)}


# ============================================================
# ACCURACY
# ============================================================


def acc(scored, mazeset=None, step=None):
    """(% correct, n) over a scored dict, optionally restricted to a maze set and/or step."""
    vals = np.fromiter(
        (
            v
            for k, v in scored.items()
            if (mazeset is None or k[0] in mazeset) and (step is None or k[1] == step)
        ),
        dtype=float,
    )
    if vals.size == 0:
        return (None, 0)
    return (100.0 * float(vals.mean()), int(vals.size))


def metadata(experiment):
    """Standard metadata block for a results file, mirroring the data files' convention."""
    return {
        "experiment": experiment,
        "produced_by": f"analysis/{experiment}.py",
        "models": MODELS,
        "n_mazes_total": 100,
        "n_consistent": {m: len(CONSISTENT[m]) for m in MODELS},
        "n_intersection": len(INTERSECTION),
        "scoring": {
            "run_idx": 0,
            "match": "exact per-step position vs the target's run-0 trajectory",
            "unparsed": "omitted at generation; audited in outcomes.json -> unparsed_records",
            "branch": ">=2 unvisited legal moves at the pre-move cell",
            "default_move": "alphabetically first UNVISITED legal direction (E < N < S < W)",
        },
    }


def pct(num, den=None, nd=1):
    """Percentage, rounded: pct(a, b) = round(100*a/b, nd); pct(frac) = round(100*frac, nd).
    Returns None when the denominator is zero/empty."""
    if den is not None:
        if not den:
            return None
        num = num / den
    return round(100.0 * num, nd)


def accuracy_matrix(kinds=("self", "cross"), mazeset=None):
    """Predictor x target accuracy table (%), computed with a groupby over RECORDS."""
    df = RECORDS[RECORDS.kind.isin(kinds)]
    if mazeset is not None:
        df = df[df.maze.isin(mazeset)]
    tab = df.groupby(["predictor", "target"])["correct"].mean().mul(100.0)
    return tab.unstack("target").reindex(index=MODELS, columns=MODELS)


# ============================================================
# MAZE GEOMETRY (problem-specific)
# ============================================================


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos):
    r, c = pos
    return [
        (nr, nc)
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        if 0 <= nr < ROWS and 0 <= nc < COLS
    ]


def bfs_dist(maze_id):
    """{cell: shortest-path distance from (0,0) through open passages}."""
    walls, dist = WALLS[maze_id], {(0, 0): 0}
    q = collections.deque([(0, 0)])
    while q:
        cur = q.popleft()
        for nb in neighbors(cur):
            if frozenset([cur, nb]) not in walls and nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def reachable_shortest(maze_id, n):
    """Cells whose shortest-path distance from (0,0) is exactly n."""
    return {c for c, d in bfs_dist(maze_id).items() if d == n}


def reachable_exactly(maze_id, n):
    """Cells reachable by a walk of exactly n steps (parity-aware: dist<=n, same parity)."""
    return {c for c, d in bfs_dist(maze_id).items() if d <= n and (n - d) % 2 == 0}


def legal_moves(target, maze, step):
    """Wall-respecting neighbor cells at the pre-move position of `step`."""
    traj = TRUTH[target][maze]
    if step < 1 or step >= len(traj):
        return []
    pos = tuple(traj[step - 1])
    return [nb for nb in neighbors(pos) if frozenset([pos, nb]) not in WALLS[maze]]


def unvisited_moves(target, maze, step):
    """Legal moves at `step` that lead to a not-yet-visited cell."""
    traj = TRUTH[target][maze]
    if step < 1 or step >= len(traj):
        return []
    visited = {tuple(p) for p in traj[:step]}
    return [nb for nb in legal_moves(target, maze, step) if nb not in visited]


def is_branch(target, maze, step):
    """True if the pre-move position offers >=2 unvisited legal moves (a genuine choice)."""
    return len(unvisited_moves(target, maze, step)) >= 2


_DIR = {(-1, 0): "North", (1, 0): "South", (0, 1): "East", (0, -1): "West"}


def direction(a, b):
    """Compass name for the unit move a->b (None if not adjacent)."""
    return _DIR.get((b[0] - a[0], b[1] - a[1]))


def chose_first_listed(target, maze, step):
    """True if the actual move at `step` was the alphabetically-first legal direction."""
    traj = TRUTH[target][maze]
    if step < 1 or step >= len(traj):
        return None
    a = tuple(traj[step - 1])
    legal = sorted(direction(a, nb) for nb in legal_moves(target, maze, step))
    if not legal:
        return None
    return direction(a, tuple(traj[step])) == legal[0]


def chose_first_unvisited(target, maze, step):
    """True if the actual move was the alphabetically-first UNVISITED legal direction.

    The labeling predicate for the default/atypical taxonomy: across 3,578 opportunities
    no model ever took a visited direction when an unvisited one was available, so the
    effective choice set at a decision point is the unvisited directions.
    """
    traj = [tuple(p) for p in TRUTH[target][maze]]
    pos = traj[step - 1]
    unv = sorted(direction(pos, tuple(nb)) for nb in unvisited_moves(target, maze, step))
    return direction(pos, traj[step]) == unv[0]


# ============================================================
# STATISTICS (library-backed)
# ============================================================


def entropy(counter):
    """Shannon entropy in bits of a Counter/dict of counts (None if empty)."""
    counts = np.fromiter(counter.values(), dtype=float)
    return float(_scipy_entropy(counts, base=2)) if counts.sum() else None


def pearson(xs, ys):
    """Pearson correlation, rounded to 3 dp (None if n<2, zero variance, or undefined)."""
    if len(xs) < 2 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantInputWarning)
        try:
            r = pearsonr(xs, ys)[0]
        except ConstantInputWarning:
            return None
    return None if np.isnan(r) else round(float(r), 3)


def perm_corr_p(xs, ys, n_perm=10000, seed=20260609):
    """Two-sided permutation p-value for the Pearson correlation (robust at small n)."""
    if pearson(xs, ys) is None:
        return None
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    res = permutation_test(
        (xs, ys),
        lambda a, b: pearsonr(a, b)[0],
        permutation_type="pairings",
        alternative="two-sided",
        n_resamples=n_perm,
        rng=np.random.default_rng(seed),
    )
    return round(float(res.pvalue), 4)


# ============================================================
# RAW RECORD ACCESS (for traces)
# ============================================================


def self_records(model, mode="reasoning"):
    """yield (maze, step, run_idx, record) for a model's self cell."""
    d = _load_predictions(os.path.join(DATA, "self_prediction", f"{model}_self_{mode}.json"))[
        f"{model}_self_{mode}"
    ]
    for mz, steps in d.items():
        for stp, recs in steps.items():
            for r in recs:
                yield mz, int(stp.split("_")[1]), r.get("run_idx"), r


def cross_records(predictor, target):
    """yield (maze, step, run_idx, record) for a predictor->target cross cell."""
    cell = f"{predictor}_xpred_{target}_reasoning"
    d = _load_predictions(
        os.path.join(DATA, "cross_prediction", f"{predictor}_xpred_reasoning.json")
    ).get(cell, {})
    for mz, steps in d.items():
        for stp, recs in steps.items():
            for r in recs:
                yield mz, int(stp.split("_")[1]), r.get("run_idx"), r
