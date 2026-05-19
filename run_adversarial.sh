#!/usr/bin/env bash
set -euo pipefail

PROFILES_PATH="${1:-data/profiles.jsonl}"
OUTPUT_DIR="${2:-results/adversarial}"
MEMBER_MODEL="${3:-gpt-4o-mini}"
CLINICIAN_MODEL="${4:-gpt-4o-mini}"
JUDGE_MODEL="${5:-gpt-4o}"
N_TURNS="${6:-10}"
MAX_WORKERS="${7:-50}"

ADVERSARIAL_TYPES=(pushback validation_seeking emotional_escalation dependency)

echo "=== Generating adversarial interactions ==="
echo "Profiles: $PROFILES_PATH | Output: $OUTPUT_DIR | Turns: $N_TURNS | Workers: $MAX_WORKERS"

python mindeval/scripts/generate_adversarial_interactions.py \
    --profiles_path "$PROFILES_PATH" \
    --clinician_system_template_version v0_1 \
    --clinician_model_api_params "{'model':'$CLINICIAN_MODEL','max_completion_tokens':4096}" \
    --member_model_api_params "{'model':'$MEMBER_MODEL','max_completion_tokens':4096}" \
    --n_turns "$N_TURNS" \
    --max_workers "$MAX_WORKERS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Judging each dimension ==="

for dim in "${ADVERSARIAL_TYPES[@]}"; do
    echo "Judging: $dim"
    python mindeval/scripts/generate_judgments.py \
        --interactions_path "$OUTPUT_DIR/${dim}.jsonl" \
        --judge_template_version v0_1 \
        --judge_model_api_params "{'model':'$JUDGE_MODEL','max_completion_tokens':4096}" \
        --max_workers "$MAX_WORKERS" \
        --output_path "$OUTPUT_DIR/judgments_${dim}.jsonl" \
        --scores_output_path "$OUTPUT_DIR/scores_${dim}.txt"
    echo "Scores for $dim:"
    cat "$OUTPUT_DIR/scores_${dim}.txt"
    echo ""
done

echo "=== Done. Results in $OUTPUT_DIR/ ==="
