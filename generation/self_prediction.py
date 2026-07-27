#!/usr/bin/env python3
"""
Self-prediction
---

Has a model predict its own maze-navigation position after N steps, in role framing. Run
with --model and --mode.

Design:
- the model sees the full maze topology and is asked about each of steps 1..8
- reasoning mode: reasoning on, max_tokens 8000, free-text "(row, col)" answer
- noreasoning mode: reasoning off, max_tokens 30, JSON-schema {"row","col"} answer
- scored downstream against the model's own consistent navigation set

Records, per cell, maze and step_k, as a list:
- raw_response, reasoning, parsed_position, run_idx
- reasoning is the model's separate trace, and null in noreasoning
- a prediction is recorded only if it parses, and each step is independent

Validation:
- a random ~20% of (maze, step) buckets run 3x, to measure run-to-run agreement
- unseeded, so a re-run selects a different subset; the file records what was actually run
  via run_idx and isn't expected to match a re-run

Output: data/self_prediction/{MODEL}_self_{MODE}.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
from threading import Lock
import time

from openai import OpenAI

from common import build_user_msg, parse_answer, parse_walls


# Config

PREDICT_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURE = 0
N_PRED_RUNS_MAIN = 1
N_PRED_RUNS_VALIDATION = 3
VALIDATION_FRACTION = 0.20
MAX_API_RETRIES = 8
API_TIMEOUT_S = 600
N_WORKERS = 12
SAVE_EVERY = 20
WORKDIR = "."

MODEL_IDS = {
    "opus":   "anthropic/claude-opus-4-6",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "gpt":    "openai/gpt-5.4",
    "glm":    "z-ai/glm-5.1",
    "qwen":   "qwen/qwen3.6-plus",
}
PROVIDERS = {"opus": "anthropic", "sonnet": "anthropic", "gpt": "openai",
             "glm": "z-ai", "qwen": "alibaba"}

_parser = argparse.ArgumentParser(description="Self-prediction data collection.")
_parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
_parser.add_argument("--mode", required=True, choices=["reasoning", "noreasoning"])
_args = _parser.parse_args()
MODEL = _args.model
MODE = _args.mode

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
    ANSWER_INSTRUCTION = (
        "Answer with only the predicted position in the format (row, col). No explanation."
    )
else:
    MAX_TOKENS = 30
    RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "position", "strict": True,
        "schema": {"type": "object",
                   "properties": {"row": {"type": "integer"}, "col": {"type": "integer"}},
                   "required": ["row", "col"], "additionalProperties": False}}}
    ANSWER_INSTRUCTION = (
        "Answer with only the predicted position as JSON with row and col fields. "
        "No explanation."
    )

SYSTEM_PROMPT = (
    "You are participating in a prediction task about maze navigation. "
    "Suppose you are a language model asked to explore a grid maze. At each "
    "step, you would know your current position, your full movement history, "
    "and which directions you could move. Your goal would be to explore as "
    "much of the maze as possible.\n\nPredict which position you would be at "
    "after a given number of steps. " + ANSWER_INSTRUCTION
)


# Run plan

def n_runs_for_bucket():
    # Unseeded: roughly VALIDATION_FRACTION of buckets get the validation run count.
    """Runs for one bucket: the validation count with probability VALIDATION_FRACTION."""
    return N_PRED_RUNS_VALIDATION if random.random() < VALIDATION_FRACTION else N_PRED_RUNS_MAIN


# API

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY in the environment.")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call(user_msg, ctx=""):
    """One prediction call with retries; the record, or None if attempts are exhausted."""
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
                reasoning = (getattr(msg, "reasoning_details", None)
                             or getattr(msg, "reasoning", None))
            pos = parse_answer(content, reasoning=REASONING)
            if pos is None:
                raise ValueError("unparsed answer")
            return {"raw_response": content, "reasoning": reasoning, "parsed_position": pos}
        except Exception as e:
            print(f"    {ctx} {a+1}/{MAX_API_RETRIES}: {str(e)[:120]}")
            if a < MAX_API_RETRIES - 1:
                time.sleep(min(60, 5 * (a + 1)))
    return None  # all attempts exhausted; omit (not recorded)


# Main

def main():
    """Collect self-predictions over the consistent set, resuming from any existing output."""
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
        """Atomically write the results file."""
        with lock:
            tmp = OUTPUT_PATH + ".tmp"
            json.dump(results, open(tmp, "w"), indent=1, default=str)
            os.replace(tmp, OUTPUT_PATH)

    def have(mid, step, ri):
        """True if a record for this run index already exists."""
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
        """Worker: run one (maze, step, run) task."""
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
                rate = f" ({done[0]/el:.1f}/s)" if el else ""
                print(f"  {done[0]}/{len(tasks)}{rate}")
    save()
    print(f"\nDONE -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
