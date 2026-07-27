#!/usr/bin/env python3
"""
Self-prediction
===============
A model predicts its OWN maze-navigation position after N steps. The model is
shown the full maze topology (every cell and its available directions) and
asked, in role framing, where it would be after each of steps 1..8.

Edit the two CONFIG lines to choose model and mode. Output filename, model id,
and metadata are derived from them.

Modes:
  reasoning    reasoning ON,  max_tokens 8000, free-text "(row, col)" answer
  noreasoning  reasoning OFF, max_tokens 30,   JSON-schema {"row","col"} answer

Records (per cell -> maze -> step_k -> list):
  raw_response, reasoning, parsed_position, run_idx
  - reasoning is the model's separate reasoning trace (null in noreasoning).
  - A prediction is recorded only if it parses; unparsed/failed attempts are
    omitted (not recorded). Each step is independent.

Validation: a random ~20% of (maze, step) buckets are run 3x (run_idx
0,1,2) to measure run-to-run agreement; the rest run once (run_idx 0). The
subset is chosen stochastically at runtime with no fixed seed, so re-running
selects a different subset by design -- the data file records what was actually
run via run_idx; it is not expected to match a re-run.

Scored downstream against the model's own consistent navigation set
(data/navigation/{model}_navigation.json, field `consistent`).

Output: data/self_prediction/{MODEL}_self_{MODE}.json
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
# CONFIG  -- edit these two lines
# ============================================================
MODEL = "opus"          # one of: opus, sonnet, gpt, glm, qwen
MODE  = "reasoning"     # one of: reasoning, noreasoning

ROWS, COLS = 5, 5
PREDICT_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURE = 0
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

assert MODEL in MODEL_IDS and MODE in ("reasoning", "noreasoning")
MODEL_ID = MODEL_IDS[MODEL]
PROVIDER = PROVIDERS[MODEL]
REASONING = (MODE == "reasoning")
CELL = f"{MODEL}_self_{MODE}"

NAV_FILE    = os.path.join(WORKDIR, "data", "navigation", f"{MODEL}_navigation.json")
OUT_DIR     = os.path.join(WORKDIR, "data", "self_prediction")
OUTPUT_PATH = os.path.join(OUT_DIR, f"{MODEL}_self_{MODE}.json")

if REASONING:
    MAX_TOKENS = 8000
    RESPONSE_FORMAT = None
    ANSWER_INSTRUCTION = "Answer with only the predicted position in the format (row, col). No explanation."
else:
    MAX_TOKENS = 30
    RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "position", "strict": True,
        "schema": {"type": "object", "properties": {"row": {"type": "integer"}, "col": {"type": "integer"}},
                   "required": ["row", "col"], "additionalProperties": False}}}
    ANSWER_INSTRUCTION = "Answer with only the predicted position as JSON with row and col fields. No explanation."

SYSTEM_PROMPT = (
    "You are participating in a prediction task about maze navigation. "
    "Suppose you are a language model asked to explore a grid maze. At each "
    "step, you would know your current position, your full movement history, "
    "and which directions you could move. Your goal would be to explore as "
    "much of the maze as possible.\n\nPredict which position you would be at "
    "after a given number of steps. " + ANSWER_INSTRUCTION
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
    if not REASONING:
        try:
            o = json.loads(content); r, c = int(o["row"]), int(o["col"])
            if 0 <= r < ROWS and 0 <= c < COLS:
                return [r, c]
        except Exception:
            pass
    ms = re.findall(r"\((\d+)\s*,\s*(\d+)\)", content)
    if ms:
        r, c = int(ms[-1][0]), int(ms[-1][1])
        if 0 <= r < ROWS and 0 <= c < COLS:
            return [r, c]
    return None

def n_runs_for_bucket():
    # Stochastic, unseeded: ~VALIDATION_FRACTION of buckets get the validation
    # run count, the rest get the main count. Not reproducible by design.
    return N_PRED_RUNS_VALIDATION if random.random() < VALIDATION_FRACTION else N_PRED_RUNS_MAIN

# ============================================================
# API
# ============================================================
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY in env or Colab userdata.")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call(user_msg, ctx=""):
    kw = dict(model=MODEL_ID,
              messages=[{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}],
              max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
              extra_body={"reasoning": {"enabled": REASONING},
                          "provider": {"only": [PROVIDER], "allow_fallbacks": False}},
              timeout=API_TIMEOUT_S)
    if RESPONSE_FORMAT:
        kw["response_format"] = RESPONSE_FORMAT
    for a in range(MAX_API_RETRIES):
        try:
            resp = client.chat.completions.create(**kw)
            msg = resp.choices[0].message
            content = msg.content
            if not content:
                raise ValueError("empty content")
            content = content.strip()
            reasoning = None
            if REASONING:
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
    nav = json.load(open(NAV_FILE))
    walls_by_id = {m["id"]: parse_walls(m["walls"]) for m in nav["mazes"]}
    maze_ids = sorted(nav["metadata"]["consistent"], key=lambda x: int(x.split("_")[1]))
    print(f"{MODEL} self-prediction [{MODE}] on {len(maze_ids)} consistent mazes\n")

    results = None
    if os.path.exists(OUTPUT_PATH):
        try: results = json.load(open(OUTPUT_PATH))
        except Exception: results = None
    if results is None:
        results = {"metadata": {
                        "experiment": f"{MODEL}_self_{MODE}",
                        "nav_source": f"{MODEL}_navigation.json",
                        "model": MODEL,
                        "model_id": MODEL_ID,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "n_pred_runs_main": N_PRED_RUNS_MAIN,
                        "n_pred_runs_validation": N_PRED_RUNS_VALIDATION,
                        "validation_fraction": VALIDATION_FRACTION,
                        "reasoning": REASONING,
                        "predict_steps": PREDICT_STEPS,
                        "maze_ids": maze_ids,
                        "cells": {CELL: {"system_prompt": SYSTEM_PROMPT}},
                   },
                   "predictions": {CELL: {}}}
    cell = results["predictions"].setdefault(CELL, {})

    lock = Lock()
    def save():
        with lock:
            tmp = OUTPUT_PATH + ".tmp"
            json.dump(results, open(tmp, "w"), indent=1, default=str)
            os.replace(tmp, OUTPUT_PATH)

    def have(mid, step, ri):
        try:
            return any(r.get("run_idx") == ri for r in cell[mid][f"step_{step}"])
        except (KeyError, TypeError):
            return False

    tasks = []
    for mid in maze_ids:
        cell.setdefault(mid, {})
        for step in PREDICT_STEPS:
            cell[mid].setdefault(f"step_{step}", [])
            for ri in range(n_runs_for_bucket()):
                if not have(mid, step, ri):
                    tasks.append((mid, step, ri))
    print(f"To run: {len(tasks)}")
    if not tasks:
        print("Nothing to do."); return

    def run_one(t):
        mid, step, ri = t
        rec = call(build_user_msg(walls_by_id[mid], step), ctx=f"{mid}|s{step}|r{ri}")
        if rec is not None:
            rec["run_idx"] = ri
        return (mid, step, ri, rec)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        done = [0]; start = time.time()
        for fut in as_completed(futs):
            mid, step, ri, rec = fut.result()
            with lock:
                if rec is not None:
                    arr = cell[mid][f"step_{step}"]
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
