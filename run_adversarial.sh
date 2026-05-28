#!/usr/bin/env bash
set -euo pipefail

PROFILES_PATH="${1:-data/profiles.jsonl}"
CLINICIAN_MODEL="${2:-gpt-4o-mini}"
MEMBER_MODEL="${3:-gpt-4o-mini}"
JUDGE_MODEL="${4:-gpt-4o}"
JUDGE_VERSION="${5:-v0_1}"
N_TURNS="${6:-10}"
MAX_WORKERS="${7:-50}"

CLINICIAN_SLUG=$(echo "$CLINICIAN_MODEL" | tr -d '-.')
INTERACTIONS_DIR="results/adversarial/${CLINICIAN_SLUG}_clinician/interactions"
OUTPUT_DIR="results/adversarial/${CLINICIAN_SLUG}_clinician/${JUDGE_VERSION}_judge"

ADVERSARIAL_TYPES=(pushback validation_seeking emotional_escalation dependency)

echo "=== Generating adversarial interactions ==="
echo "Profiles: $PROFILES_PATH | Clinician: $CLINICIAN_MODEL | Member: $MEMBER_MODEL | Turns: $N_TURNS | Workers: $MAX_WORKERS"

mkdir -p "$INTERACTIONS_DIR"

python mindeval/scripts/generate_adversarial_interactions.py \
    --profiles_path "$PROFILES_PATH" \
    --clinician_system_template_version v0_1 \
    --clinician_model_api_params "{'model':'$CLINICIAN_MODEL','max_completion_tokens':4096}" \
    --member_model_api_params "{'model':'$MEMBER_MODEL','max_completion_tokens':4096}" \
    --n_turns "$N_TURNS" \
    --max_workers "$MAX_WORKERS" \
    --output_dir "$INTERACTIONS_DIR"

echo ""
echo "=== Judging each dimension ==="

mkdir -p "$OUTPUT_DIR"

for dim in "${ADVERSARIAL_TYPES[@]}"; do
    echo "Judging: $dim"
    python mindeval/scripts/generate_judgments.py \
        --interactions_path "$INTERACTIONS_DIR/${dim}.jsonl" \
        --judge_template_version "$JUDGE_VERSION" \
        --judge_model_api_params "{'model':'$JUDGE_MODEL','max_completion_tokens':4096}" \
        --max_workers "$MAX_WORKERS" \
        --output_path "$OUTPUT_DIR/judgments_${dim}.jsonl" \
        --scores_output_path "$OUTPUT_DIR/scores_${dim}.txt"
    echo "Scores for $dim:"
    cat "$OUTPUT_DIR/scores_${dim}.txt"
    echo ""
done

echo "=== Done. Interactions in $INTERACTIONS_DIR/ | Results in $OUTPUT_DIR/ ==="
