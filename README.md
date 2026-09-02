# llm-self-prediction

A behavioral experiment testing whether a model predicts its own 
behavior better than other models predict it.

## Experiment

5 models navigated the same 100 mazes, then predicted each other's routes. This produced a 5x5 predictor-target matrix whose diagonal is
self-prediction accuracy and whose off-diagonal is the control. 

Data was collected in 3 phases for 5 models at temperature 0:

| name | model id |
|---|---|
| opus | `anthropic/claude-opus-4-6` |
| sonnet | `anthropic/claude-sonnet-4-6` |
| gpt | `openai/gpt-5.4` |
| glm | `z-ai/glm-5.1` |
| qwen | `qwen/qwen3.6-plus` |

**Navigation** — Mazes were cyclic 5x5 grids, each a DFS spanning tree 
with 6 walls removed to create loops and branch points. Models walked an 8-step route 
3 times with reasoning off. A maze was only scored for a model if all 3 runs matched.

**Self-prediction** — A model saw a full maze and predicted its own position 
after every step, in 2 modes: reasoning off with a 30-token JSON `{"row", "col"}` 
answer, and reasoning on with a free-text `(row, col)` answer.

**Cross-prediction** — Every model predicted the others' positions, scored against 
the target's consistent maze set. This was done only with reasoning enabled, since self-prediction
accuracy collapsed without reasoning.

## Directory

```
data/
  navigation/                maze definitions and model navigation runs
  self_prediction/           {model}_self_{reasoning,noreasoning}.json
  cross_prediction/          {predictor}_xpred_reasoning.json, one cell per target
  self_framing_pilot.json    prompt framing test
generation/                  collection scripts and the shared prompt + parsing helpers
analysis/                    analysis pipeline
analysis/results/            one JSON per analysis script
```

## Reproduction

To generate a new dataset, install `openai`, add `OPENROUTER_API_KEY` to the environment, and run from the repository root:

```bash
python3 generation/navigation.py --model opus
python3 generation/self_prediction.py --model opus --mode reasoning
python3 generation/cross_prediction.py --predictor opus
python3 generation/self_framing_pilot.py
```

Every bootstrap and permutation test is seeded, so all results files are reproducible on Python 3.12. `run_all.py` runs every script in order and stops at any failure.

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python analysis/run_all.py
```

## Validation

`analysis/validate_substitution.py` re-runs the entire pipeline three times, substituting run 
1 or run 2 for run 0 wherever an alternate record exists. Recomputed headline quantities are
written alongside originals to `analysis/results/validation_substitution.json`.

```bash
venv/bin/python analysis/validate_substitution.py
```

Alternate runs cover 2,902 of 14,640 predictions (19.8%). Within these, the models mostly repeat 
themselves: only 1.50% (run 1) and 1.82% (run 2) of scored positions change.
