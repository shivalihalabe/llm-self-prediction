#!/usr/bin/env python3
"""
Shared scoring contract and record table
---

Loads every prediction and navigation file once and exposes the scored records plus the
derived views the analysis scripts read. Paths resolve relative to this file, so scripts can
be run from anywhere.

Scoring contract:
- run_idx == 0 records only; runs 1 and 2 are reserved for the noise-floor analysis
- per-step exact position match against the target's run-0 navigation trajectory
- unparsed records, where parsed_position is None, are skipped
- maze-set treatments: native (the target's consistent set), 5-way intersection, pairwise

Core objects:
- RECORDS: one row per scored run-0 prediction, with columns kind, predictor, target, maze,
  step, pred, truth, correct
- SELF[m], SELF_NR[m], CROSS[(p, t)]: {(maze, step): correct}
- SELF_POS[m], SELF_NR_POS[m], CROSS_POS[(p, t)]: {(maze, step): [row, col]}
- CONSISTENT[m], INTERSECTION, PAIRWISE[(a, b)], MAZE_DIFFICULTY, DIFFICULTY_STRATA
- TRUTH[m][maze]: the run-0 trajectory
- WALLS[maze]: the blocked cell pairs
"""

import collections
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy as _scipy_entropy, permutation_test, ConstantInputWarning

ROWS = COLS = 5
MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "..", "data")  # analysis/ sits at the repo root next to data/


# Navigation ground truth

def _nav(model):
    """Parsed navigation file for a model."""
    return json.load(open(os.path.join(DATA, "navigation", f"{model}_navigation.json")))


TRUTH = {
    m: {mz: node["runs"][0]["trajectory"] for mz, node in _nav(m)["navigation"][m].items()}
    for m in MODELS
}
WALLS = {
    mz["id"]: {frozenset([tuple(a), tuple(b)]) for a, b in mz["walls"]}
    for mz in _nav(MODELS[0])["mazes"]
}
for _m in MODELS[1:]:
    _w = {
        mz["id"]: {frozenset([tuple(a), tuple(b)]) for a, b in mz["walls"]}
        for mz in _nav(_m)["mazes"]
    }
    assert _w == WALLS, f"maze definitions in {_m}'s navigation file differ from {MODELS[0]}'s"
del _w


# Build the record table


# RUN_PREF selects which prediction run supplies the scored position. It exists for the
# run-substitution check in validate_substitution.py and is 0, the committed behaviour, unless
# the environment sets it. A cell without a parsed record for the preferred run falls back to
# run 0, so the scored set is the same size under every setting and the variants stay
# like-for-like; only about a fifth of cells carry an alternate run at all.
RUN_PREF = int(os.environ.get("RUN_PREF", "0"))


def _preferred_run(payload):
    """The record for the preferred run, falling back to run 0 when it has no parsed answer."""
    if RUN_PREF:
        rec = next(
            (
                r
                for r in payload
                if r.get("run_idx") == RUN_PREF and r.get("parsed_position") is not None
            ),
            None,
        )
        if rec is not None:
            return rec
    return next((r for r in payload if r.get("run_idx") == 0), None)


def _iter_scored(node, target):
    """Yield (maze, step, pred_tuple, correct) for run-0 parsed records under a prediction node."""
    truth = TRUTH[target]
    for mz, steps in node.items():
        if mz not in truth:
            continue
        for stp, payload in steps.items():
            si = int(stp.split("_")[1])
            if si >= len(truth[mz]):
                continue
            rec = _preferred_run(payload)
            if not rec or rec.get("parsed_position") is None:
                continue
            pred = tuple(rec["parsed_position"])
            yield mz, si, pred, pred == tuple(truth[mz][si])


def _load_predictions(path):
    """The predictions block of a prediction file."""
    return json.load(open(path))["predictions"]


def _records():
    """Build the record rows for the self, no-reasoning and cross cells."""
    rows = []
    for m in MODELS:
        for mode, kind in [("reasoning", "self"), ("noreasoning", "self_nr")]:
            node = _load_predictions(
                os.path.join(DATA, "self_prediction", f"{m}_self_{mode}.json")
            )[f"{m}_self_{mode}"]
            for mz, si, pred, ok in _iter_scored(node, m):
                rows.append((kind, m, m, mz, si, pred, ok))
    for p in MODELS:
        xp = _load_predictions(os.path.join(DATA, "cross_prediction", f"{p}_xpred_reasoning.json"))
        for t in MODELS:
            cell = f"{p}_xpred_{t}_reasoning"
            if cell not in xp:
                continue
            for mz, si, pred, ok in _iter_scored(xp[cell], t):
                rows.append(("cross", p, t, mz, si, pred, ok))
    df = pd.DataFrame(
        rows, columns=["kind", "predictor", "target", "maze", "step", "pred", "correct"]
    )
    df["truth"] = [tuple(TRUTH[t][mz][s]) for t, mz, s in zip(df.target, df.maze, df.step)]
    return df


RECORDS = _records()


# Dict views

def _as_dicts(frame):
    """Split a record frame into (correct-by-cell, predicted-position-by-cell)."""
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


# Maze sets

CONSISTENT = {
    m: set(RECORDS[(RECORDS.kind == "self") & (RECORDS.target == m)].maze.unique()) for m in MODELS
}
# The consistent set is inferred from which mazes appear in the self-prediction records; check
# it against the navigation data (three runs, identical trajectories) and require the full
# eight scored steps per maze, so a cell silently dropped at generation would surface here.
for _m in MODELS:
    _runs_by_maze = _nav(_m)["navigation"][_m]
    for _mz in CONSISTENT[_m]:
        _runs = [r["trajectory"] for r in _runs_by_maze[_mz]["runs"]]
        assert len(_runs) == 3 and _runs[0] == _runs[1] == _runs[2], (
            f"{_m}/{_mz}: consistent maze lacks three identical navigation runs"
        )
        _n_scored = sum((_mz, _s) in SELF[_m] for _s in range(1, 9))
        assert _n_scored == 8, f"{_m}/{_mz}: {_n_scored} scored steps, expected 8"
INTERSECTION = set.intersection(*(CONSISTENT[m] for m in MODELS))
PAIRWISE = {
    tuple(sorted((a, b))): CONSISTENT[a] & CONSISTENT[b]
    for i, a in enumerate(MODELS)
    for b in MODELS[i + 1 :]
}

_ALL = set().union(*CONSISTENT.values())
MAZE_DIFFICULTY = {mz: sum(mz in CONSISTENT[m] for m in MODELS) for mz in _ALL}
DIFFICULTY_STRATA = {k: {mz for mz, c in MAZE_DIFFICULTY.items() if c == k} for k in range(1, 6)}


# Accuracy

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
        "run_pref": RUN_PREF,
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


# Maze geometry

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos):
    """In-bounds grid neighbours of pos, ignoring walls."""
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
    """Cells whose shortest-path distance from (0,0) is n."""
    return {c for c, d in bfs_dist(maze_id).items() if d == n}


def reachable_exactly(maze_id, n):
    """Cells reachable by a walk of exactly n steps (parity-aware: dist<=n, same parity)."""
    return {c for c, d in bfs_dist(maze_id).items() if d <= n and (n - d) % 2 == 0}


def legal_moves(target, maze, step):
    """Wall-respecting neighbor cells at the pre-move position of step."""
    traj = TRUTH[target][maze]
    if step < 1 or step >= len(traj):
        return []
    pos = tuple(traj[step - 1])
    return [nb for nb in neighbors(pos) if frozenset([pos, nb]) not in WALLS[maze]]


def unvisited_moves(target, maze, step):
    """Legal moves at step that lead to a not-yet-visited cell."""
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


def chose_first_unvisited(target, maze, step):
    """True if the actual move was the alphabetically-first unvisited legal direction.

    The canonical predicate defining the default/atypical split: across 3,578 opportunities
    no model ever took a visited direction when an unvisited one was available, so the
    effective choice set at a decision point is the unvisited directions. A superseded
    definition (first-listed among all legal directions) was removed rather than kept as
    a comparator.
    """
    traj = [tuple(p) for p in TRUTH[target][maze]]
    if step < 1 or step >= len(traj):
        return None
    pos = traj[step - 1]
    unv = sorted(direction(pos, tuple(nb)) for nb in unvisited_moves(target, maze, step))
    return direction(pos, traj[step]) == unv[0]


# Statistics

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


def modal_position(positions):
    """The most common position, ties broken by count descending then position ascending.

    Counter.most_common resolves ties by insertion order, so it depends on row order; this
    doesn't.
    """
    counts = collections.Counter(positions)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def generic_answer_cells(predictor):
    """Cells where a predictor answered about itself and about at least two other targets.

    Returns (maze, step, self_pred, generic_pred, self_correct, generic_correct) per cell.
    The generic answer is the modal position the predictor gave about the other targets on
    that maze and step, and both answers are scored against the predictor's own run-0
    trajectory, so the two are comparable. Ties in the mode break by count descending then
    position ascending, so the result doesn't depend on row order.
    """
    sub = RECORDS[RECORDS.kind.isin(("self", "cross")) & (RECORDS.predictor == predictor)]
    cells = {}
    for r in sub.itertuples(index=False):
        cells.setdefault((r.maze, r.step), []).append(r)
    out = []
    for (mz, step), rows in sorted(cells.items()):
        own = [r for r in rows if r.target == predictor]
        others = [r for r in rows if r.target != predictor]
        if len(own) != 1 or len(others) < 2:
            continue
        generic = modal_position(r.pred for r in others)
        row = own[0]
        out.append((mz, step, row.pred, generic, bool(row.correct), generic == row.truth))
    return out


def best_other_model(t, mazeset=None):
    """The external model with the highest cross-accuracy on target t (optional maze set)."""
    cand = {p: acc(CROSS[(p, t)], mazeset)[0] for p in MODELS if p != t and (p, t) in CROSS}
    cand = {p: v for p, v in cand.items() if v is not None}
    return max(cand, key=cand.get) if cand else None


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
    return round(float(res.pvalue), 6)


def fmt_p(p, n_draws):
    """Emission format for a p-value estimated from n_draws permutation draws.

    Such a test cannot resolve below one draw in n_draws, so 1/n_draws is the floor and a
    returned zero means only that no draw reached the observed statistic. Flooring rather
    than reporting the zero keeps every emitted p-value an upper bound on the true one.
    """
    if p is None:
        return None
    return round(max(float(p), 1.0 / n_draws), 6)


# Raw record access

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
