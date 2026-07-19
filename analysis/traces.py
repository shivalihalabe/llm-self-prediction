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
# WRITE + SUMMARY
# ============================================================

with open(os.path.join(OUT, "traces.json"), "w") as f:
    json.dump(RES, f, indent=1)

if __name__ == "__main__":
    print("first-person 'I' correct vs wrong  - RAW count then PER-100-WORDS rate:")
    print(f"  {'model':7} {'raw_cor':>8} {'raw_wro':>8}   {'rate_cor':>9} {'rate_wro':>9}")
    for m, d in by_model.items():
        x = d["lexical"]["I"]
        print(
            f"  {m:7} {x['raw_correct']:>8} {x['raw_wrong']:>8}   {x['per100w_correct']:>9} {x['per100w_wrong']:>9}"
        )
    print("\nhedging (per-100-words, correct vs wrong) - length-robust uncertainty signal:")
    for m, d in by_model.items():
        h = d["lexical"]["hedging"]
        print(f"  {m:7} correct={h['per100w_correct']}  wrong={h['per100w_wrong']}")
    print("\nunique positions (structural, correct vs wrong) + coherence:")
    for m, d in by_model.items():
        u = d["structural"]["unique_positions"]
        print(
            f"  {m:7} correct={u['correct']:>5} wrong={u['wrong']:>5}  coherence={d['trace_answer_coherence_pct']}%"
        )
    print("\nself vs cross 'I' - raw then per-100-words:")
    for p, d in selfcross.items():
        print(
            f"  {p:7} raw self={d['self_I_raw']} cross={d['cross_I_raw']}   per100w self={d['self_I_per100w']} cross={d['cross_I_per100w']}"
        )
    print("\ntrace-path simulation - fraction of true path walked in order (correct vs wrong):")
    for m, d in sim.items():
        print(
            f"  {m:7} correct={d['true_path_tracked_correct']}  wrong={d['true_path_tracked_wrong']}"
        )
    print("\nlength dose-response - accuracy by trace-length tercile:")
    for m, d in lenacc.items():
        if d:
            print(
                f"  {m:7} short={d['short']['acc']}% (n={d['short']['n']})  medium={d['medium']['acc']}%  long={d['long']['acc']}%"
            )
    print("\ntrace features by step (opus) - length / unique-pos / truth-never-appears%:")
    for r in features_by_step["opus"]:
        print(
            f"  step {r['step']}: len={str(r['len']):>6} unique_pos={str(r['unique_pos']):>5} never_appears={str(r['truth_never_appears_pct']) + '%' if r['truth_never_appears_pct'] is not None else 'n/a'} (n_wrong={r['n_wrong']})"
        )
    print("\nhedging as a confidence signal - accuracy when hedging vs not:")
    for m, d in hedging_cal.items():
        print(
            f"  {m:7} hedging {d['acc_when_hedging']}% (n={d['n_hedging']})  not {d['acc_when_not_hedging']}% (n={d['n_plain']})"
        )
    print("\n-> wrote results/traces.json")
