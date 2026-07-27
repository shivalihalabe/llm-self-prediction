#!/usr/bin/env python3
"""
Reasoning-trace analysis
========================

Processes the `reasoning` field of each run-0 prediction, correlating trace features with
correctness per model. Lexical markers are reported BOTH raw (absolute count per trace) and
length-normalized (per 100 words), because raw counts track trace length: a model whose wrong
traces are simply longer will show more of every marker. The per-100-word rate isolates whether
a marker is genuinely enriched in wrong (vs correct) traces independent of length.

Each trace is featurized once into a tidy per-trace frame (lexical counts, structural features,
coordinate chronology, true-path prefix); every analysis below is an aggregation over that frame.

Markers: first-person "I", "we", LLM-framing, hedging, reconsidering, decisiveness, search
mentions, determinism mentions, step-by-step markers. Structural features (length, word count,
arrow density, digit/letter ratio, unique grid positions referenced) are reported as-is.
Also: chronology of the truth in wrong traces, lock-in point, trace-answer coherence, and a
self-vs-cross depersonalization comparison (both raw and normalized).

The `reasoning` field is a string, a list of {"text": ...} dicts, or None -- handled by
trace_text().
Output: analysis/results/traces.json
"""

import json
import os
import re
import collections
import statistics as st

import pandas as pd

import common as C

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
MODELS = C.MODELS
RES = {"metadata": C.metadata("traces")}

_COORD = re.compile(r"\((\d)\s*,\s*(\d)\)")
_PATTERNS = {
    "I": re.compile(r"\bI\b"),
    "we": re.compile(r"\bwe\b", re.I),
    "llm_framing": re.compile(r"\blanguage model\b|\ban?\s+AI\b|\bthe model\b", re.I),
    "hedging": re.compile(r"\b(probably|likely|maybe|might|perhaps|possibly)\b", re.I),
    "reconsider": re.compile(r"\b(wait|actually|reconsider|hmm|recheck|re-check)\b", re.I),
    "decisive": re.compile(r"\b(clearly|definitely|certainly|obviously|surely)\b", re.I),
    "search_terms": re.compile(r"\b(BFS|DFS|backtrack|depth-first|breadth-first)\b", re.I),
    "determinism": re.compile(r"\b(temperature|deterministic|determinism|stochastic)\b", re.I),
    "step_marker": re.compile(r"\bstep\s*\d", re.I),
}
LEXICAL = list(_PATTERNS)  # reported raw + per-100-words
STRUCT = ["len", "word_count", "arrow_per_1000c", "digit_letter_ratio", "unique_positions"]


def trace_text(rec):
    r = rec.get("reasoning")
    if r is None:
        return ""
    if isinstance(r, str):
        return r
    if isinstance(r, list):
        return "\n".join(
            (e.get("text") or e.get("content") or "") if isinstance(e, dict) else str(e) for e in r
        )
    return str(r)


def coords_with_pos(text):
    """ordered [( (r,c), char_index )] for in-bounds coordinates mentioned in the trace."""
    out = []
    for m in _COORD.finditer(text):
        r, c = int(m.group(1)), int(m.group(2))
        if 0 <= r < C.ROWS and 0 <= c < C.COLS:
            out.append(((r, c), m.start()))
    return out


def longest_true_prefix(coords, true_path):
    """How many positions of true_path are found, in order, within the trace's coordinate list."""
    ti = 0
    for pos in coords:
        if ti < len(true_path) and pos == tuple(true_path[ti]):
            ti += 1
            if ti >= len(true_path):
                break
    return ti


def _mean(vals):
    vals = list(vals)
    return round(st.mean(vals), 2) if vals else None


def _rate_per_100w(count, words):
    return (100.0 * count / words) if words else 0.0


# ============================================================
# FEATURIZE EVERY RUN-0 SELF TRACE ONCE
# ============================================================


def _featurize(m, mz, s, rec):
    text = trace_text(rec)
    if not text.strip():
        return {"model": m, "maze": mz, "step": s, "is_empty": True}
    L = len(text)
    words = len(text.split())
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    cw = coords_with_pos(text)
    pred = tuple(rec["parsed_position"])
    valid = mz in C.TRUTH[m] and s < len(C.TRUTH[m][mz])
    truth = tuple(C.TRUTH[m][mz][s]) if valid else None
    first_pred = next((pos for c, pos in cw if c == pred), None)
    first_truth = next((pos for c, pos in cw if c == truth), None) if truth is not None else None
    row = {
        "model": m,
        "maze": mz,
        "step": s,
        "is_empty": False,
        "valid": valid,
        "correct": bool(truth is not None and pred == truth),
        "len": L,
        "word_count": words,
        "arrow_per_1000c": round(1000.0 * text.count("->") / L, 2) if L else 0,
        "digit_letter_ratio": round(digits / letters, 3) if letters else 0,
        "unique_positions": len(set(c for c, _ in cw)),
        "coherent": bool(cw and cw[-1][0] == pred),
        "lockin": round(first_pred / L, 3) if first_pred is not None and L else None,
        "hedges": bool(_PATTERNS["hedging"].search(text)),
        "prefix_frac": (
            (
                longest_true_prefix(
                    [c for c, _ in cw], [tuple(C.TRUTH[m][mz][i]) for i in range(0, s + 1)]
                )
                / (s + 1)
            )
            if valid
            else None
        ),
    }
    if truth is not None and pred != truth:
        if first_truth is None:
            row["chrono"] = "truth_never_appears"
        elif first_pred is None or first_truth < first_pred:
            row["chrono"] = "truth_before_pred"
        else:
            row["chrono"] = "pred_before_truth"
    else:
        row["chrono"] = None
    for name, pat in _PATTERNS.items():
        n = len(pat.findall(text))
        row[name] = n
        row[name + "_per100w"] = _rate_per_100w(n, words)
    return row


TRACES = pd.DataFrame(
    [
        _featurize(m, mz, s, rec)
        for m in MODELS
        for mz, s, ri, rec in C.self_records(m, "reasoning")
        if ri == 0 and rec.get("parsed_position") is not None
    ]
)
for _col in ("is_empty", "valid", "correct", "coherent", "hedges"):
    TRACES[_col] = TRACES[_col].fillna(False).astype(bool)


# ============================================================
# PER-MODEL: FEATURES BY CORRECTNESS
# ============================================================

by_model = {}
for m in MODELS:
    sub = TRACES[(TRACES.model == m) & ~TRACES.is_empty]
    cor, wro = sub[sub.correct], sub[~sub.correct]
    by_model[m] = {
        "n_traces": int(len(sub)),
        "n_empty_skipped": int(TRACES[TRACES.model == m].is_empty.sum()),
        "lexical": {
            k: {
                "raw_correct": _mean(cor[k]),
                "raw_wrong": _mean(wro[k]),
                "per100w_correct": _mean(cor[k + "_per100w"]),
                "per100w_wrong": _mean(wro[k + "_per100w"]),
            }
            for k in LEXICAL
        },
        "structural": {k: {"correct": _mean(cor[k]), "wrong": _mean(wro[k])} for k in STRUCT},
        "chronology_when_wrong": dict(collections.Counter(sub.chrono.dropna())),
        "lockin_fraction": {
            "correct": _mean(cor.lockin.dropna()),
            "wrong": _mean(wro.lockin.dropna()),
        },
        "trace_answer_coherence_pct": C.pct(sub.coherent.mean()) if len(sub) else None,
    }
RES["self_traces"] = by_model


# ============================================================
# SELF VS CROSS: DEPERSONALIZATION (raw + normalized)
# ============================================================


def _collect(record_iter):
    I_raw, I_rate, llm_raw, llm_rate, n = [], [], [], [], 0
    for mz, s, ri, rec in record_iter:
        if ri != 0:
            continue
        text = trace_text(rec)
        if not text.strip():
            continue
        n += 1
        w = len(text.split())
        ic = len(_PATTERNS["I"].findall(text))
        lc = len(_PATTERNS["llm_framing"].findall(text))
        I_raw.append(ic)
        I_rate.append(_rate_per_100w(ic, w))
        llm_raw.append(lc)
        llm_rate.append(_rate_per_100w(lc, w))
    return I_raw, I_rate, llm_raw, llm_rate, n


selfcross = {}
for p in MODELS:
    sI, sIr, sL, sLr, sn = _collect(C.self_records(p, "reasoning"))
    cI, cIr, cL, cLr = [], [], [], []
    cn = 0
    for t in MODELS:
        if t == p or (p, t) not in C.CROSS:
            continue
        a, b, c, d, k = _collect(C.cross_records(p, t))
        cI += a
        cIr += b
        cL += c
        cLr += d
        cn += k
    selfcross[p] = {
        "self_I_raw": _mean(sI),
        "cross_I_raw": _mean(cI),
        "self_I_per100w": _mean(sIr),
        "cross_I_per100w": _mean(cIr),
        "self_llm_raw": _mean(sL),
        "cross_llm_raw": _mean(cL),
        "self_llm_per100w": _mean(sLr),
        "cross_llm_per100w": _mean(cLr),
        "n_self": sn,
        "n_cross": cn,
    }
RES["self_vs_cross_depersonalization"] = selfcross


# ============================================================
# TRACE-PATH SIMULATION + LENGTH DOSE-RESPONSE
# ============================================================

# Does the trace actually walk the true trajectory (reason-like-you-navigate), and does accuracy
# depend on how long the model reasons?
sim, lenacc = {}, {}
for m in MODELS:
    sub = TRACES[(TRACES.model == m) & ~TRACES.is_empty & TRACES.valid]
    sim[m] = {
        "true_path_tracked_correct": _mean(sub[sub.correct].prefix_frac),
        "true_path_tracked_wrong": _mean(sub[~sub.correct].prefix_frac),
    }
    ld = (
        sub[["len", "correct"]]
        .sort_values(["len", "correct"], kind="stable")
        .reset_index(drop=True)
    )
    terc = {}
    if len(ld):
        n = len(ld)
        for name, seg in [
            ("short", ld.iloc[: n // 3]),
            ("medium", ld.iloc[n // 3 : 2 * n // 3]),
            ("long", ld.iloc[2 * n // 3 :]),
        ]:
            if len(seg):
                terc[name] = {
                    "n": int(len(seg)),
                    "acc": C.pct(seg.correct.mean()),
                    "char_len_range": [int(seg.len.iloc[0]), int(seg.len.iloc[-1])],
                }
    lenacc[m] = terc
RES["trace_path_simulation"] = sim
RES["length_accuracy"] = lenacc


# ============================================================
# TRACE FEATURES BY STEP (horizon-resolved)
# ============================================================

features_by_step = {}
for m in MODELS:
    sub = TRACES[(TRACES.model == m) & ~TRACES.is_empty & TRACES.step.between(1, 8)]
    rows = []
    for s in range(1, 9):
        ss = sub[sub.step == s]
        wrong = ss[ss.valid & ~ss.correct]
        tot = int(len(wrong))
        rows.append(
            {
                "step": s,
                "len": _mean(ss.len),
                "unique_pos": _mean(ss.unique_positions),
                "reconsider_per100w": _mean(ss.reconsider_per100w),
                "truth_never_appears_pct": (
                    C.pct((wrong.chrono == "truth_never_appears").mean()) if tot else None
                ),
                "n_wrong": tot,
            }
        )
    features_by_step[m] = rows
RES["features_by_step"] = features_by_step


# ============================================================
# IS HEDGING A USABLE CONFIDENCE SIGNAL?
# ============================================================

# Within each model, split run-0 traces by whether they contain any hedging word; compare accuracy.
hedging_cal = {}
for m in MODELS:
    sub = TRACES[(TRACES.model == m) & ~TRACES.is_empty & TRACES.valid]
    h, pl = sub[sub.hedges], sub[~sub.hedges]
    hedging_cal[m] = {
        "acc_when_hedging": C.pct(h.correct.mean()) if len(h) else None,
        "n_hedging": int(len(h)),
        "acc_when_not_hedging": C.pct(pl.correct.mean()) if len(pl) else None,
        "n_plain": int(len(pl)),
    }
RES["hedging_calibration"] = hedging_cal


# ============================================================
# DIRECTION LANGUAGE AT ATYPICAL CELLS
# ============================================================

# Which compass words a model's reasoning uses on its own atypical cells -- a lexical view of the
# self-model (e.g. a model whose atypical moves are southward while its traces talk about "east").
_DIRWORDS = {d: re.compile(rf"\b{d}\b", re.I) for d in ("north", "south", "east", "west")}
dirlang = {}
for m in MODELS:
    at = {
        (mz, s)
        for (mz, s) in C.SELF[m]
        if C.is_branch(m, mz, s) and not C.chose_first_unvisited(m, mz, s)
    }
    sub = TRACES[(TRACES.model == m) & ~TRACES.is_empty]
    sub = sub[[k in at for k in zip(sub.maze, sub.step)]]
    n = int(len(sub))
    counts = {d: 0 for d in _DIRWORDS}
    for mz, s, ri, rec in C.self_records(m, "reasoning"):
        if ri != 0 or (mz, s) not in at or rec.get("parsed_position") is None:
            continue
        txt = trace_text(rec)
        if not txt.strip():
            continue
        for d, pat in _DIRWORDS.items():
            counts[d] += bool(pat.search(txt))
    dirlang[m] = {"n_traces": n, "pct_mentioning": {d: C.pct(c, n) for d, c in counts.items()}}
RES["atypical_cell_direction_language"] = dirlang


# ============================================================
# WORDS PER BRANCH POINT VS ACCURACY
# ============================================================

# Reasoning length normalised by difficulty: words in the trace divided by the number of genuine
# branch decisions in the predicted path (cells with >=1 branch). Terciles replace the raw
# character-length split, whose gradient is a difficulty confound.
wpb = {}
for m in MODELS:
    rows = []
    for mz, s, ri, rec in C.self_records(m, "reasoning"):
        if ri != 0 or rec.get("parsed_position") is None:
            continue
        txt = trace_text(rec)
        if not txt.strip() or mz not in C.TRUTH[m] or s >= len(C.TRUTH[m][mz]):
            continue
        nb = sum(1 for i in range(1, s + 1) if C.is_branch(m, mz, i))
        if nb == 0:
            continue
        rows.append(
            (len(txt.split()) / nb, tuple(rec["parsed_position"]) == tuple(C.TRUTH[m][mz][s]))
        )
    rows.sort(key=lambda x: (x[0], x[1]))
    n = len(rows)
    terc = {}
    for name, seg in (
        ("short", rows[: n // 3]),
        ("medium", rows[n // 3 : 2 * n // 3]),
        ("long", rows[2 * n // 3 :]),
    ):
        if seg:
            terc[name] = {
                "n": len(seg),
                "acc": C.pct(sum(okv for _, okv in seg), len(seg)),
                "ratio_range": [round(seg[0][0], 1), round(seg[-1][0], 1)],
            }
    wpb[m] = terc
RES["words_per_branch_terciles"] = wpb


# ============================================================
# WRITE
# ============================================================

with open(os.path.join(OUT, "traces.json"), "w") as f:
    json.dump(RES, f, indent=1)
