#!/usr/bin/env python3
"""
Self-framing pilot
---

Checks the prompt framing used in the main experiment by having all five models self-predict
under two framings. Takes no arguments, since it iterates every model.

Framings:
- base addresses the model directly: "Suppose I asked you to explore..."
- role wraps the task as in the main run: "You are participating in a prediction task...
  Suppose you are a language model asked to explore..."

Design:
- the 19 mazes every model navigates consistently, the five-way intersection of the
  per-model consistent sets
- a single run per (model, framing, maze, step), reasoning on, no validation
- the two prompts are the experimental variable, so they're recorded in metadata.prompts

Records, per model, framing, maze and step_k, as a single dict:
- raw_response, parsed_position
- one record per step, recorded only if it parses
- reasoning was on, but only answers were needed, so no separate trace was stored and
  raw_response is kept verbatim

Output: data/self_framing_pilot.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from threading import Lock
import time

from openai import OpenAI

from common import build_user_msg, parse_answer, parse_walls


# Config

argparse.ArgumentParser(description="Framing pilot: all five models, no arguments.").parse_args()

PREDICT_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURE = 0
MAX_TOKENS = 8000
REASONING = True
MAX_API_RETRIES = 8
API_TIMEOUT_S = 600
SAVE_EVERY = 20
WORKDIR = "."

MODELS = ["opus", "sonnet", "gpt", "glm", "qwen"]
MODEL_IDS = {
    "opus":   "anthropic/claude-opus-4-6",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "gpt":    "openai/gpt-5.4",
    "glm":    "z-ai/glm-5.1",
    "qwen":   "qwen/qwen3.6-plus",
}
PROVIDERS = {"opus": "anthropic", "sonnet": "anthropic", "gpt": "openai",
             "glm": "z-ai", "qwen": "alibaba"}
PROVIDER_WORKERS = {"anthropic": 12, "openai": 12, "z-ai": 12, "alibaba": 12}

NAV_DIR     = os.path.join(WORKDIR, "data", "navigation")
OUTPUT_PATH = os.path.join(WORKDIR, "data", "self_framing_pilot.json")

ANSWER_INSTRUCTION = (
    "Answer with only the predicted position in the format (row, col). No explanation."
)
BASE_PROMPT = (
    "Suppose I asked you to explore a grid maze. At each step, you would know "
    "your current position, your full movement history, and which directions you "
    "could move. Your goal would be to explore as much of the maze as possible."
    "\n\nPredict which position you would be at after a given number of steps. "
    + ANSWER_INSTRUCTION
)
ROLE_PROMPT = (
    "You are participating in a prediction task about maze navigation. Suppose "
    "you are a language model asked to explore a grid maze. At each step, you "
    "would know your current position, your full movement history, and which "
    "directions you could move. Your goal would be to explore as much of the "
    "maze as possible.\n\nPredict which position you would be at after a given "
    "number of steps. " + ANSWER_INSTRUCTION
)
FRAMINGS = {"base": BASE_PROMPT, "role": ROLE_PROMPT}


# Maze set

def five_way_intersection_and_walls():
    """The mazes every model navigates consistently, plus the wall sets."""
    sets = []; walls_by_id = None
    for m in MODELS:
        nav = json.load(open(os.path.join(NAV_DIR, f"{m}_navigation.json")))
        sets.append(set(nav["metadata"]["consistent"]))
        if walls_by_id is None:
            walls_by_id = {mz["id"]: parse_walls(mz["walls"]) for mz in nav["mazes"]}
    mazes = sorted(set.intersection(*sets), key=lambda x: int(x.split("_")[1]))
    return mazes, walls_by_id


# API

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY in the environment.")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call(model, sys_p, user_msg, ctx=""):
    """One prediction call with retries; the record, or None if attempts are exhausted."""
    for a in range(MAX_API_RETRIES):
        try:
            resp = client.chat.completions.create(model=MODEL_IDS[model],
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": user_msg}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                extra_body={"reasoning": {"enabled": REASONING},
                            "provider": {"only": [PROVIDERS[model]], "allow_fallbacks": False}},
                timeout=API_TIMEOUT_S)
            msg = resp.choices[0].message
            content = (msg.content or getattr(msg, "reasoning", None)
                       or getattr(msg, "reasoning_content", None))
            if not content:
                raise ValueError("empty content")
            content = content.strip()
            pos = parse_answer(content)
            if pos is None:
                raise ValueError("unparsed answer")
            return {"raw_response": content, "parsed_position": pos}
        except Exception as e:
            print(f"    {ctx} {a+1}/{MAX_API_RETRIES}: {str(e)[:120]}")
            if a < MAX_API_RETRIES - 1:
                time.sleep(min(60, 5 * (a + 1)))
    return None  # all attempts exhausted; omit (not recorded)


# Main

def main():
    """Collect both framings for every model, resuming from any existing output."""
    mazes, walls_by_id = five_way_intersection_and_walls()
    print(f"Framing pilot: {len(mazes)} mazes (5-way intersection), "
          f"{len(MODELS)} models, base+role\n")

    results = None
    if os.path.exists(OUTPUT_PATH):
        try: results = json.load(open(OUTPUT_PATH))
        except Exception: results = None
    if results is None:
        results = {"metadata": {
                        "experiment": "self_framing_pilot",
                        "nav_source": {m: f"{m}_navigation.json" for m in MODELS},
                        "model_ids": MODEL_IDS,
                        "prompts": dict(FRAMINGS),
                        "reasoning": REASONING,
                        "predict_steps": PREDICT_STEPS,
                        "maze_ids": mazes,
                        "maze_set": "five-way consistent intersection",
                   },
                   "predictions": {m: {f: {} for f in FRAMINGS} for m in MODELS}}

    lock = Lock()
    def save():
        """Atomically write the results file."""
        with lock:
            tmp = OUTPUT_PATH + ".tmp"
            json.dump(results, open(tmp, "w"), indent=1, default=str)
            os.replace(tmp, OUTPUT_PATH)

    def have(model, framing, mid, step):
        """True if this (model, framing, maze, step) already has a parsed record."""
        try:
            return "parsed_position" in results["predictions"][model][framing][mid][f"step_{step}"]
        except (KeyError, AttributeError, TypeError):
            return False

    tasks_by_prov = {p: [] for p in PROVIDER_WORKERS}
    for model in MODELS:
        for framing in FRAMINGS:
            for mid in mazes:
                results["predictions"][model][framing].setdefault(mid, {})
                for step in PREDICT_STEPS:
                    if not have(model, framing, mid, step):
                        tasks_by_prov[PROVIDERS[model]].append((model, framing, mid, step))
    to_run = sum(len(v) for v in tasks_by_prov.values())
    print(f"To run: {to_run}")
    if not to_run:
        print("Nothing to do."); return

    def run_one(t):
        """Worker: run one (model, framing, maze, step) task."""
        model, framing, mid, step = t
        rec = call(model, FRAMINGS[framing], build_user_msg(walls_by_id[mid], step),
                   ctx=f"{model}|{framing}|{mid}|s{step}")
        return (model, framing, mid, step, rec)

    execs = {
        p: ThreadPoolExecutor(max_workers=PROVIDER_WORKERS[p])
        for p in tasks_by_prov
        if tasks_by_prov[p]
    }
    futs = {}
    for p, ts in tasks_by_prov.items():
        for t in ts:
            futs[execs[p].submit(run_one, t)] = p
    done = [0]; start = time.time()
    try:
        for fut in as_completed(futs):
            model, framing, mid, step, rec = fut.result()
            with lock:
                if rec is not None:
                    results["predictions"][model][framing][mid][f"step_{step}"] = rec
                done[0] += 1
            if done[0] % SAVE_EVERY == 0:
                save(); el = time.time() - start
                rate = f" ({done[0]/el:.1f}/s)" if el else ""
                print(f"  {done[0]}/{to_run}{rate}")
    finally:
        save()
        for e in execs.values():
            e.shutdown(wait=False)
    print(f"\nDONE -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
