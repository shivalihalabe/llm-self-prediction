#!/usr/bin/env python3
"""
Cross-prediction
================
A PREDICTOR model predicts a TARGET model's maze-navigation position after N
steps. The predictor sees the full maze topology and is asked, in role framing
that names the target, where the target would be after each of steps 1..8. One
file per predictor, with a cell for each of the other four targets. Each target
is predicted on THAT target's consistent navigation set, so cell sizes differ.

Edit PREDICTOR to switch. Output filename, model id, and metadata derive from it.

Mode note: cross-prediction is run with reasoning ON only. Without reasoning the
task collapses to a near-constant prior, so there is no no-reasoning cross
matrix; the no-reasoning baseline is self-prediction only. (The `_reasoning`
suffix is kept on the file and cells so they read in parallel with the self
files, not because a noreasoning counterpart exists.)

Records (per cell -> maze -> step_k -> list):
  raw_response, reasoning, parsed_position, run_idx
  - reasoning is the predictor's separate reasoning trace.
  - A prediction is recorded only if it parses; unparsed/failed attempts are
    omitted (not recorded).

Validation: a random ~20% of (cell, maze, step) buckets are run 3x
(run_idx 0,1,2); the rest run once. Stochastic, unseeded -- the subset is
not reproducible on a re-run by design; the data records it via run_idx.

Output: data/cross_prediction/{PREDICTOR}_xpred_reasoning.json
"""

import os
import random
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from google.colab import userdata, files
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
    IS_COLAB = True
except Exception:
    IS_COLAB = False
    files = None

from openai import OpenAI

# ============================================================
# CONFIG  -- edit this one line
# ============================================================
PREDICTOR = "opus"   # one of: opus, sonnet, gpt, glm, qwen

ROWS, COLS = 5, 5
PREDICT_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURE = 0
MAX_TOKENS = 8000
REASONING = True
N_PRED_RUNS_MAIN = 1
N_PRED_RUNS_VALIDATION = 3
VALIDATION_FRACTION = 0.20
MAX_API_RETRIES = 8
API_TIMEOUT_S = 600
N_WORKERS = 12
SAVE_EVERY = 20
WORKDIR = "/content" if IS_COLAB else "."

MODEL_IDS = {
    "opus":   "anthropic/claude-opus-4-6",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "gpt":    "openai/gpt-5.4",
    "glm":    "z-ai/glm-5.1",
    "qwen":   "qwen/qwen3.6-plus",
}
PROVIDERS = {"opus": "anthropic", "sonnet": "anthropic", "gpt": "openai",
             "glm": "z-ai", "qwen": "alibaba"}
DISPLAY = {"opus": "Claude Opus 4.6", "sonnet": "Claude Sonnet 4.6", "gpt": "GPT-5.4",
           "glm": "GLM 5.1", "qwen": "Qwen 3.6 Plus"}
ALL_MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]

assert PREDICTOR in MODEL_IDS
MODEL_ID = MODEL_IDS[PREDICTOR]
PROVIDER = PROVIDERS[PREDICTOR]
TARGETS = [m for m in ALL_MODELS if m != PREDICTOR]
ANSWER_INSTRUCTION = "Answer with only the predicted position in the format (row, col). No explanation."

NAV_DIR     = os.path.join(WORKDIR, "data", "navigation")
OUT_DIR     = os.path.join(WORKDIR, "data", "cross_prediction")
OUTPUT_PATH = os.path.join(OUT_DIR, f"{PREDICTOR}_xpred_reasoning.json")

def system_prompt(target):
    name = DISPLAY[target]
    return (
        "You are participating in a prediction task about maze navigation. "
        f"Suppose {name} was asked to explore a grid maze. At each step, {name} "
        "would know its current position, its full movement history, and which "
        "directions it could move. Its goal would be to explore as much of the "
        f"maze as possible.\n\nPredict which position {name} would be at after a "
        f"given number of steps. " + ANSWER_INSTRUCTION
    )

# ============================================================
# MAZE / PROMPT
# ============================================================
def parse_walls(wl):
    return set(frozenset([tuple(p[0]), tuple(p[1])]) for p in wl)

def get_available_directions(pos, walls):
    r, c = pos
    out = {}
    for d, (nr, nc) in {"North": (r-1, c), "South": (r+1, c), "East": (r, c+1), "West": (r, c-1)}.items():
        if 0 <= nr < ROWS and 0 <= nc < COLS and frozenset([pos, (nr, nc)]) not in walls:
            out[d] = (nr, nc)
    return out

def describe_maze_topology(walls):
    lines = [f"Grid maze: {ROWS}x{COLS}. Positions (row,col), (0,0) top-left.",
             "Directions from each position:"]
    for r in range(ROWS):
        for c in range(COLS):
            dirs = get_available_directions((r, c), walls)
            if dirs:
                lines.append(f"  ({r},{c}): " + ", ".join(f"{n}->({a},{b})" for n, (a, b) in sorted(dirs.items())))
    return "\n".join(lines)

def build_user_msg(walls, n_steps):
    return f"{describe_maze_topology(walls)}\n\nStarting at (0, 0), predict the position after {n_steps} steps."

def parse_answer(content):
    if not content:
        return None
    ms = re.findall(r"\((\d+)\s*,\s*(\d+)\)", content)
    if ms:
        r, c = int(ms[-1][0]), int(ms[-1][1])
        if 0 <= r < ROWS and 0 <= c < COLS:
            return [r, c]
    return None

def n_runs_for_bucket():
    return N_PRED_RUNS_VALIDATION if random.random() < VALIDATION_FRACTION else N_PRED_RUNS_MAIN

# ============================================================
# API
# ============================================================
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY in env or Colab userdata.")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call(sys_p, user_msg, ctx=""):
    for a in range(MAX_API_RETRIES):
        try:
            resp = client.chat.completions.create(model=MODEL_ID,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": user_msg}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                extra_body={"reasoning": {"enabled": REASONING},
                            "provider": {"only": [PROVIDER], "allow_fallbacks": False}},
                timeout=API_TIMEOUT_S)
            msg = resp.choices[0].message
            content = msg.content
            if not content:
                raise ValueError("empty content")
            content = content.strip()
            reasoning = getattr(msg, "reasoning_details", None) or getattr(msg, "reasoning", None)
            pos = parse_answer(content)
            if pos is None:
                raise ValueError("unparsed answer")
            return {"raw_response": content, "reasoning": reasoning, "parsed_position": pos}
        except Exception as e:
            print(f"    {ctx} {a+1}/{MAX_API_RETRIES}: {str(e)[:120]}")
            if a < MAX_API_RETRIES - 1:
                time.sleep(min(60, 5 * (a + 1)))
    return None  # all attempts exhausted; omit (not recorded)

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    target_walls = {}; target_mazes = {}
    for t in TARGETS:
        nav = json.load(open(os.path.join(NAV_DIR, f"{t}_navigation.json")))
        target_walls[t] = {m["id"]: parse_walls(m["walls"]) for m in nav["mazes"]}
        target_mazes[t] = sorted(nav["metadata"]["consistent"], key=lambda x: int(x.split("_")[1]))
    print(f"{PREDICTOR} cross-prediction (reasoning) of {TARGETS}")
    print("  target sizes:", {t: len(target_mazes[t]) for t in TARGETS}, "\n")

    results = None
    if os.path.exists(OUTPUT_PATH):
        try: results = json.load(open(OUTPUT_PATH))
        except Exception: results = None
    if results is None:
        cells = {}
        for t in TARGETS:
            cells[f"{PREDICTOR}_xpred_{t}_reasoning"] = {
                "nav_source": f"{t}_navigation.json",
                "target": t,
                "target_model_id": MODEL_IDS[t],
                "target_model_name_in_prompt": DISPLAY[t],
                "maze_ids": target_mazes[t],
                "system_prompt": system_prompt(t),
            }
        results = {"metadata": {
                        "experiment": f"{PREDICTOR}_xpred_reasoning",
                        "predictor": PREDICTOR,
                        "predictor_model_id": MODEL_ID,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "n_pred_runs_main": N_PRED_RUNS_MAIN,
                        "n_pred_runs_validation": N_PRED_RUNS_VALIDATION,
                        "validation_fraction": VALIDATION_FRACTION,
                        "reasoning": REASONING,
                        "predict_steps": PREDICT_STEPS,
                        "cells": cells,
                   },
                   "predictions": {}}

    lock = Lock()
    def save():
        with lock:
            tmp = OUTPUT_PATH + ".tmp"
            json.dump(results, open(tmp, "w"), indent=1, default=str)
            os.replace(tmp, OUTPUT_PATH)

    def have(cell, mid, step, ri):
        try:
            return any(r.get("run_idx") == ri
                       for r in results["predictions"][cell][mid][f"step_{step}"])
        except (KeyError, TypeError):
            return False

    tasks = []
    for t in TARGETS:
        cell = f"{PREDICTOR}_xpred_{t}_reasoning"
        results["predictions"].setdefault(cell, {})
        for mid in target_mazes[t]:
            results["predictions"][cell].setdefault(mid, {})
            for step in PREDICT_STEPS:
                results["predictions"][cell][mid].setdefault(f"step_{step}", [])
                for ri in range(n_runs_for_bucket()):
                    if not have(cell, mid, step, ri):
                        tasks.append((t, cell, mid, step, ri))
    print(f"To run: {len(tasks)}")
    if not tasks:
        print("Nothing to do."); return

    def run_one(task):
        t, cell, mid, step, ri = task
        rec = call(system_prompt(t), build_user_msg(target_walls[t][mid], step), ctx=f"{cell}|{mid}|s{step}|r{ri}")
        if rec is not None:
            rec["run_idx"] = ri
        return (cell, mid, step, ri, rec)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        done = [0]; start = time.time()
        for fut in as_completed(futs):
            cell, mid, step, ri, rec = fut.result()
            with lock:
                if rec is not None:
                    arr = results["predictions"][cell][mid][f"step_{step}"]
                    arr[:] = [r for r in arr if r.get("run_idx") != ri] + [rec]
                done[0] += 1
            if done[0] % SAVE_EVERY == 0:
                save(); el = time.time() - start
                print(f"  {done[0]}/{len(tasks)} ({done[0]/el:.1f}/s)" if el else f"  {done[0]}/{len(tasks)}")
    save()
    print(f"\nDONE -> {OUTPUT_PATH}")
    if IS_COLAB and files is not None:
        try: files.download(OUTPUT_PATH)
        except Exception: pass

if __name__ == "__main__":
    main()
