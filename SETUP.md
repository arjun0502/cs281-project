# Setup Guide

## Prerequisites

- Python 3.9–3.12 (tested on 3.12.3)
- [Poetry](https://python-poetry.org/docs/#installation) 2.1.4+

## 1. Clone the repo

```bash
git clone https://github.com/arjun0502/cs281-project
cd cs281-project
```

## 2. Create and activate the virtual environment

```bash
python -m venv .cs281_project_venv
source .cs281_project_venv/bin/activate
```

## 3. Install dependencies

```bash
poetry install
```

## 4. Configure API credentials

Create a `.env` file at the project root (already gitignored) with your OpenAI key:

```bash
# .env
OPENAI_API_KEY=sk-...
```

Load it before running scripts:

```bash
set -a && source .env && set +a
```

## 5. Smoke test (run this first)

Before running the full benchmark, verify everything works end-to-end on a small subset. Create a 3-profile test file:

```bash
head -3 data/profiles.jsonl > data/profiles_test.jsonl
```

Run interactions:

```bash
python mindeval/scripts/generate_interactions.py \
    --profiles_path data/profiles_test.jsonl \
    --clinician_system_template_version v0_1 \
    --clinician_model_api_params "{'model':'gpt-4o-mini','max_completion_tokens':4096}" \
    --member_system_template_version v0_2 \
    --member_model_api_params "{'model':'gpt-4o-mini','max_completion_tokens':4096}" \
    --n_turns 10 \
    --max_workers 50 \
    --output_path interactions_test.jsonl \
    --text_output_path interactions_test.txt
```

Run judgments:

```bash
python mindeval/scripts/generate_judgments.py \
    --interactions_path interactions_test.jsonl \
    --judge_template_version v0_1 \
    --judge_model_api_params "{'model':'gpt-4o','max_completion_tokens':4096}" \
    --max_workers 50 \
    --output_path judgments_test.jsonl
```

Check scores:

```bash
python -c "
import pandas as pd
from mindeval.utils import load_jsonl
judgments = load_jsonl('judgments_test.jsonl')
df = pd.DataFrame([j['parsed_judgment'] for j in judgments])
print(df)
"
```

If you see a table of scores, everything is working.

## 6. Run the full baseline

```bash
python mindeval/scripts/generate_interactions.py \
    --profiles_path data/profiles.jsonl \
    --clinician_system_template_version v0_1 \
    --clinician_model_api_params "{'model':'gpt-4o-mini','max_completion_tokens':4096}" \
    --member_system_template_version v0_2 \
    --member_model_api_params "{'model':'gpt-4o-mini','max_completion_tokens':4096}" \
    --n_turns 10 \
    --max_workers 50 \
    --output_path interactions_baseline.jsonl

python mindeval/scripts/generate_judgments.py \
    --interactions_path interactions_baseline.jsonl \
    --judge_template_version v0_1 \
    --judge_model_api_params "{'model':'gpt-4o','max_completion_tokens':4096}" \
    --max_workers 50 \
    --output_path judgments_baseline.jsonl
```
