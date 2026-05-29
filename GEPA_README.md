# GEPA Clinician Prompt Optimization

Iteratively improves the AI clinician system prompt across 5 evaluation dimensions using the GEPA framework.

## Setup

```bash
# 1. Activate the project virtualenv
source ~/Library/Caches/pypoetry/virtualenvs/mindeval-N8VL5211-py3.11/bin/activate

# 2. Load API key
set -a && source .env && set +a

# 3. Install GEPA (one-time)
pip install gepa
```

## Running

```bash
# From the project root (cs281-project/)
python -m pipeline.main
```

## If Data Moves

Update the two path constants in [pipeline/config.py](pipeline/config.py):

```python
INTERACTIONS_DIR = "results/adversarial/gpt4omini_clinician/interactions"
JUDGMENTS_DIR    = "results/adversarial/gpt4omini_clinician/v0_3_judge"
```

## Smoke Test (no API credits)

```bash
python pipeline/tests/smoke.py
```

Verifies all imports, data loading, simulation formatting, judge parsing, and metric wiring without making any real API calls.

## Outputs

| Path | Contents |
|------|----------|
| `outputs/best_prompt_{dimension}.txt` | Best accepted prompt after each dimension |
| `outputs/final_clinician_prompt.txt` | Final prompt after all cycles |
| `outputs/run_log.txt` | Full run log with scores and accept/reject decisions |
| `outputs/rejected_{dimension}_cycle{n}.txt` | Prompts that caused regression (saved for inspection) |
| `gepa_runs/cycle_{n}/{dimension}/` | GEPA internal run artifacts |

## Architecture

```
pipeline/
  config.py      — all constants (paths, models, hyperparameters)
  data_prep.py   — load 200 adversarial conversations, build per-dimension trainsets
  simulate.py    — multi-turn conversation simulation (mirrors generate_interactions.py)
  metric.py      — run_judge() + make_metric() for GEPA evaluators
  regression.py  — held-out regression check after each dimension
  utils.py       — logging and prompt file I/O
  main.py        — optimization loop: 5 dimensions × up to 3 cycles
  tests/
    smoke.py     — zero-credit verification of the full pipeline
```

## Dimension Optimization Order

1. `therapeutic_alliance` → Therapeutic Relationship & Alliance
2. `ai_communication`     → AI-Specific Communication Quality
3. `clinical_accuracy`    → Clinical Accuracy & Competence
4. `assessment_response`  → Assessment & Response
5. `ethical_conduct`      → Ethical & Professional Conduct

Each dimension's best prompt seeds the next. After all 5, the cycle repeats up to 3 times or until improvement plateaus (< 0.05 average gain per cycle).
