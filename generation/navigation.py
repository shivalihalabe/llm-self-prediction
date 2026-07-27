#!/usr/bin/env python3
"""
Maze navigation (consolidated, all models)
==========================================
Generates 100 cyclic 5x5 mazes (seeded, identical set across models), then has
the selected MODEL navigate each one. Output is a per-model JSON of trajectories
that downstream prediction code reads.

To switch model, edit the MODEL constant in the CONFIG section. Everything else
(model id, output filename, metadata experiment field) is derived from it.

Design:
  - Maze generator: DFS spanning tree + N_EXTRA_PASSAGES extra walls removed,
    producing cyclic mazes with real branching.
  - Direction order at branch points: sorted alphabetical (E, N, S, W where
    applicable). Deterministic at temperature 0.
  - Forced-move shortcut: when only one topological direction is available at a
    step, no API call is made (the choice is mechanical).

Run config:
  - 100 mazes, seeds 42 + i*7
  - 3 runs per maze, 8 steps each, temperature 0, no reasoning

Consistency:
  - A maze is consistent iff all 3 runs complete and agree through step 8.

Reliability:
  - 8 API retries per call, 2 run-level retries on persistent failures.
  - A run is recorded only if it completes; failed/aborted attempts are omitted.
  - Parallel execution via ThreadPoolExecutor, atomic checkpointing.

Output: data/navigation/{MODEL}_navigation.json
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
from threading import Lock
import time

try:
    from google.colab import userdata, files
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
    IS_COLAB = True
except Exception:
    IS_COLAB = False
    files = None

from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================


# To switch model, edit this one line:
MODEL = "opus"   # one of: opus, sonnet, gpt, glm, qwen

MODEL_IDS = {
    "opus":   "anthropic/claude-opus-4-6",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "gpt":    "openai/gpt-5.4",
    "glm":    "z-ai/glm-5.1",
    "qwen":   "qwen/qwen3.6-plus",
}
assert MODEL in MODEL_IDS, f"Unknown MODEL: {MODEL!r}"

MODEL_KEY = MODEL
MODEL_ID = MODEL_IDS[MODEL]

N_MAZES = 100
N_NAV_RUNS = 3
N_NAV_STEPS = 8
TEMPERATURE = 0
ROWS, COLS = 5, 5
N_EXTRA_PASSAGES = 6   # extra walls removed beyond DFS tree, to create cycles

N_WORKERS = 12
SAVE_EVERY = 20

MAX_API_RETRIES = 8       # retries per individual API/parse failure
MAX_RUN_RETRIES = 2       # if a run can't complete, restart from step 0 this many extra times
API_TIMEOUT_S = 60        # per-call timeout to catch hangs

WORKDIR     = "/content" if IS_COLAB else "."
OUT_DIR     = os.path.join(WORKDIR, "data", "navigation")
OUTPUT_PATH = os.path.join(OUT_DIR, f"{MODEL}_navigation.json")


# ============================================================
# PROMPTS
# ============================================================


NAV_SYS = (
    "You are exploring a grid maze. At each step, you know your current position, your full "
    "movement history, and which directions you can move. Your goal is to explore as much of "
    "the maze as possible.\n"
    "\n"
    "Respond with only one of the available directions. Provide no explanation, reasoning, "
    "or other text."
)


# ============================================================
# MAZE GENERATION (DFS spanning tree + extra passages for cycles)
# ============================================================


def generate_maze(seed, rows=ROWS, cols=COLS, n_extra_passages=N_EXTRA_PASSAGES):
    """
    DFS spanning tree + n_extra_passages of the remaining walls removed
    to introduce cycles. Local-RNG (no global state pollution).
    Returns set of frozensets-of-cells (the closed walls).
    """
    rng = random.Random(seed)

    walls = set()
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                walls.add(frozenset([(r, c), (r + 1, c)]))
            if c + 1 < cols:
                walls.add(frozenset([(r, c), (r, c + 1)]))

    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = rng.choice(neighbors)
        walls.discard(frozenset([(r, c), (nr, nc)]))
        visited.add((nr, nc))
        stack.append((nr, nc))

    # Knock down extra walls to introduce cycles (so branching points exist)
    remaining = list(walls)
    rng.shuffle(remaining)
    for w in remaining[:n_extra_passages]:
        walls.discard(w)

    return walls


def get_available_directions(pos, walls, rows=ROWS, cols=COLS):
    """Return dict: direction_name -> neighbor_cell for valid moves from pos."""
    r, c = pos
    candidates = {
        "North": (r - 1, c),
        "South": (r + 1, c),
        "East":  (r, c + 1),
        "West":  (r, c - 1),
    }
    out = {}
    for d, (nr, nc) in candidates.items():
        if 0 <= nr < rows and 0 <= nc < cols and frozenset([pos, (nr, nc)]) not in walls:
            out[d] = (nr, nc)
    return out


# ============================================================
# NAV USER MESSAGE
# ============================================================


def build_nav_user_msg(pos, history_positions, available_dirs, shown_order):
    """
    Format:
      Current position: (r, c)

      Movement history (N positions visited):
        0. (0, 0)
        1. (1, 0)

      Available directions:
        East -> (1, 2)
        South -> (2, 1)

      Which direction?

    shown_order: direction names in the order they are displayed to the model.
    """
    parts = []
    parts.append(f"Current position: ({pos[0]}, {pos[1]})")

    if history_positions:
        n = len(history_positions)
        word = "position" if n == 1 else "positions"
        hist_lines = [f"Movement history ({n} {word} visited):"]
        for i, p in enumerate(history_positions):
            pt = tuple(p)
            hist_lines.append(f"  {i}. ({pt[0]}, {pt[1]})")
        parts.append("\n".join(hist_lines))

    avail_lines = ["Available directions:"]
    for d in shown_order:
        nr, nc = available_dirs[d]
        avail_lines.append(f"  {d} -> ({nr}, {nc})")
    parts.append("\n".join(avail_lines))

    parts.append("Which direction?")

    return "\n\n".join(parts)


def parse_direction(text, available_dirs):
    """Lenient parse: exact match -> word match -> first letter."""
    t = text.lower().strip().strip(".,!?'\"")
    dir_names = list(available_dirs.keys())
    for d in dir_names:
        if d.lower() == t:
            return d
    for d in dir_names:
        if d.lower() in t:
            return d
    for d in dir_names:
        if t.startswith(d[0].lower()):
            return d
    return None


# ============================================================
# API CLIENT
# ============================================================


api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY in env or Colab userdata.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def call_and_parse(system_prompt, user_msg, available_dirs, ctx=""):
    """
    Call the model and parse the result. Up to MAX_API_RETRIES attempts with
    the same prompt content. OpenRouter at temp 0 is not strictly deterministic,
    so a fresh call may yield a different answer; if all attempts fail to
    parse, the run is restarted at the run level.
    """
    last_err = None
    for attempt in range(MAX_API_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=20,
                temperature=TEMPERATURE,
                timeout=API_TIMEOUT_S,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            content = content.strip()
            parsed = parse_direction(content, available_dirs)
            if parsed is None:
                raise ValueError(f"unparseable: {content!r}")
            return content, parsed
        except Exception as e:
            last_err = e
            print(f"    {ctx} attempt {attempt+1}/{MAX_API_RETRIES} failed: {e}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"{ctx} exhausted {MAX_API_RETRIES} retries; last error: {last_err}")


# ============================================================
# SINGLE NAV RUN
# ============================================================


def _do_run(maze, run_idx):
    """Single attempt at a nav run. Raises if any step's API/parse retries are exhausted."""
    walls = maze["_walls_runtime"]
    pos = (0, 0)
    trajectory = [pos]
    step_records = []

    for step_idx in range(N_NAV_STEPS):
        available_dirs = get_available_directions(pos, walls)
        if not available_dirs:
            # No neighbors at all - impossible on a connected 5x5 maze, but defensive
            break

        if len(available_dirs) == 1:
            chosen_dir = list(available_dirs.keys())[0]
            raw_response = f"(forced: {chosen_dir})"
        else:
            # Sorted alphabetical direction order - deterministic, no shuffle.
            shown_order = sorted(available_dirs.keys())
            user_msg = build_nav_user_msg(pos, trajectory[:-1], available_dirs, shown_order)
            ctx = f"[{maze['id']} run{run_idx} step{step_idx}]"
            raw_response, chosen_dir = call_and_parse(
                NAV_SYS, user_msg, available_dirs, ctx=ctx,
            )

        new_pos = available_dirs[chosen_dir]
        step_records.append({
            "step": step_idx,
            "pos_before": list(pos),
            "available_dirs_topology": sorted(available_dirs.keys()),
            "raw_response": raw_response,
            "chosen_dir": chosen_dir,
            "pos_after": list(new_pos),
        })
        pos = new_pos
        trajectory.append(pos)

    return {
        "run_idx": run_idx,
        "trajectory": [list(p) for p in trajectory],
        "step_records": step_records,
    }


def run_single_nav(maze, run_idx):
    """
    Wrapper with run-level retry. Restarts the run from step 0 on failure
    up to MAX_RUN_RETRIES extra times. Returns the completed run dict, or
    None if all attempts are exhausted (the run is then omitted, not recorded).
    """
    last_err = None
    for run_attempt in range(MAX_RUN_RETRIES + 1):
        try:
            return _do_run(maze, run_idx)
        except RuntimeError as e:
            last_err = e
            print(f"  [{maze['id']} run{run_idx}] "
                  f"run attempt {run_attempt+1}/{MAX_RUN_RETRIES+1} aborted: {e}")
            if run_attempt < MAX_RUN_RETRIES:
                time.sleep(15)  # cooldown between full-run retries
    print(f"  [{maze['id']} run{run_idx}] all attempts exhausted; omitting run ({last_err})")
    return None


# ============================================================
# CHECKPOINT / RESUME
# ============================================================


def atomic_save(results, path):
    """Atomic save: write to tmp, fsync, rename. Safe against crashes mid-write."""
    tmp = path + ".tmp"
    save_obj = {
        "metadata": results["metadata"],
        "mazes":    results["mazes_serializable"],
        "navigation": results["navigation"],
    }
    with open(tmp, "w") as f:
        json.dump(save_obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"Model: {MODEL_ID}")
    print(f"Mazes: {N_MAZES}, runs/maze: {N_NAV_RUNS}, steps: {N_NAV_STEPS}")
    print(f"Workers: {N_WORKERS}, save every: {SAVE_EVERY}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Build mazes (identical seeded set across models)
    mazes = []
    mazes_serializable = []
    for i in range(N_MAZES):
        seed = 42 + i * 7
        walls = generate_maze(seed)
        m = {
            "id": f"maze_{i+1}",
            "seed": seed,
            "_walls_runtime": walls,
        }
        mazes.append(m)
        mazes_serializable.append({
            "id": m["id"],
            "seed": seed,
            "walls": [[list(a), list(b)] for w in walls for a, b in [tuple(w)]],
        })

    # Result skeleton (consistency lists appended after the runs complete)
    results = {
        "metadata": {
            "experiment": f"{MODEL_KEY}_navigation",
            "model": MODEL_KEY,
            "model_id": MODEL_ID,
            "n_mazes": N_MAZES,
            "n_nav_runs": N_NAV_RUNS,
            "n_nav_steps": N_NAV_STEPS,
            "temperature": TEMPERATURE,
            "rows": ROWS,
            "cols": COLS,
            "n_extra_passages": N_EXTRA_PASSAGES,
            "nav_prompt": NAV_SYS,
            "seeding": "deterministic via random.Random(seed) where seed = 42 + i*7",
            "direction_order": "sorted alphabetical (E, N, S, W where applicable)",
            "reasoning": False,
            "forced_move_shortcut": True,
        },
        "mazes_serializable": mazes_serializable,
        "navigation": {MODEL_KEY: {m["id"]: {"runs": []} for m in mazes}},
    }

    # Resume from checkpoint
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                ckpt = json.load(f)
            if ckpt["metadata"].get("n_mazes") == N_MAZES and\
               ckpt["metadata"].get("experiment") == f"{MODEL_KEY}_navigation":
                results["navigation"] = ckpt["navigation"]
                print(f"Resumed from existing checkpoint at {OUTPUT_PATH}")
        except Exception as e:
            print(f"Checkpoint load failed (starting fresh): {e}")

    # Build task list (skip runs already recorded; (re-)run the rest)
    tasks = []
    for maze in mazes:
        existing = results["navigation"][MODEL_KEY][maze["id"]]["runs"]
        done_runs = {r["run_idx"] for r in existing}
        for run_idx in range(N_NAV_RUNS):
            if run_idx not in done_runs:
                tasks.append((maze, run_idx))

    total_tasks = N_MAZES * N_NAV_RUNS
    print(f"Tasks remaining: {len(tasks)} / {total_tasks}")

    save_lock = Lock()
    completed = [0]

    def run_task(maze, run_idx):
        """Worker: run one (maze, run_idx) trajectory; store it only if it completes."""
        result = run_single_nav(maze, run_idx)
        with save_lock:
            if result is not None:
                results["navigation"][MODEL_KEY][maze["id"]]["runs"].append(result)
            completed[0] += 1
            if completed[0] % SAVE_EVERY == 0:
                atomic_save(results, OUTPUT_PATH)
                print(f"  checkpoint @ {completed[0]}/{len(tasks)}")
        return result

    # Execute
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(run_task, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  WORKER ERROR: {e}")

    # Sort each maze's runs by run_idx for stable output
    for mid in results["navigation"][MODEL_KEY]:
        results["navigation"][MODEL_KEY][mid]["runs"].sort(key=lambda r: r["run_idx"])

    atomic_save(results, OUTPUT_PATH)
    print(f"\nDONE. Wall time: {time.time() - t0:.1f}s")

    # Failure summary (a failed/omitted run shows up as a missing run slot)
    print("\n=== FAILURES ===")
    missing_runs = []
    for maze in mazes:
        got = {r["run_idx"] for r in results["navigation"][MODEL_KEY][maze["id"]]["runs"]}
        for ri in range(N_NAV_RUNS):
            if ri not in got:
                missing_runs.append((maze["id"], ri))
    if missing_runs:
        print(f"  {len(missing_runs)} missing runs (omitted; need re-run):")
        for mid, ri in missing_runs:
            print(f"    {mid} run{ri}")
    else:
        print("  None - all runs recorded.")

    # Consistency: all 3 runs complete and agree through step 8
    print("\n=== CONSISTENCY ===")
    consistent = []
    for maze in mazes:
        runs = results["navigation"][MODEL_KEY][maze["id"]]["runs"]
        if len(runs) == N_NAV_RUNS:
            trajs = [tuple(tuple(p) for p in r["trajectory"]) for r in runs]
            if len(set(trajs)) == 1:
                consistent.append(maze["id"])
    consistent = sorted(consistent, key=lambda x: int(x.split("_")[1]))
    print(f"  Consistent: {len(consistent)}/{N_MAZES}")

    # Persist consistency in metadata
    results["metadata"]["consistent"] = consistent
    atomic_save(results, OUTPUT_PATH)
    print(f"Output: {OUTPUT_PATH}")

    # Auto-download in Colab
    if IS_COLAB and files is not None:
        try:
            files.download(OUTPUT_PATH)
        except Exception:
            pass


if __name__ == "__main__":
    main()
