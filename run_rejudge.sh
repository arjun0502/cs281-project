#!/usr/bin/env bash
set -euo pipefail

INTERACTIONS_DIR="${1:-results/adversarial_gpt4omini_clinician_v0_1_judge}"
OUTPUT_DIR="${2:-results/adversarial_gpt4omini_clinician_v0_2_judge}"
BASELINE_OUTPUT_DIR="${3:-results/baseline_gpt4omini_clinician_v0_2_judge}"
MEMBER_MODEL="${4:-gpt-4o-mini}"
CLINICIAN_MODEL="${5:-gpt-4o-mini}"
JUDGE_MODEL="${6:-gpt-4o}"
N_TURNS="${7:-10}"
MAX_WORKERS="${8:-50}"

ADVERSARIAL_TYPES=(pushback validation_seeking emotional_escalation dependency)

echo "=== Regenerating baseline interactions ==="
mkdir -p "$BASELINE_OUTPUT_DIR"

python mindeval/scripts/generate_interactions.py \
    --profiles_path data/profiles.jsonl \
    --clinician_system_template_version v0_1 \
    --clinician_model_api_params "{'model':'$CLINICIAN_MODEL','max_completion_tokens':4096}" \
    --member_system_template_version v0_2 \
    --member_model_api_params "{'model':'$MEMBER_MODEL','max_completion_tokens':4096}" \
    --n_turns "$N_TURNS" \
    --max_workers "$MAX_WORKERS" \
    --output_path "$BASELINE_OUTPUT_DIR/interactions_baseline.jsonl"

echo "=== Judging baseline with v0_2 judge ==="

python mindeval/scripts/generate_judgments.py \
    --interactions_path "$BASELINE_OUTPUT_DIR/interactions_baseline.jsonl" \
    --judge_template_version v0_2 \
    --judge_model_api_params "{'model':'$JUDGE_MODEL','max_completion_tokens':4096}" \
    --max_workers "$MAX_WORKERS" \
    --output_path "$BASELINE_OUTPUT_DIR/judgments_baseline.jsonl" \
    --scores_output_path "$BASELINE_OUTPUT_DIR/scores_baseline.txt"

echo "Baseline scores:"
cat "$BASELINE_OUTPUT_DIR/scores_baseline.txt"
echo ""

echo "=== Re-judging adversarial interactions with v0_2 judge ==="
echo "Interactions: $INTERACTIONS_DIR | Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

for dim in "${ADVERSARIAL_TYPES[@]}"; do
    echo "Judging: $dim"
    python mindeval/scripts/generate_judgments.py \
        --interactions_path "$INTERACTIONS_DIR/${dim}.jsonl" \
        --judge_template_version v0_2 \
        --judge_model_api_params "{'model':'$JUDGE_MODEL','max_completion_tokens':4096}" \
        --max_workers "$MAX_WORKERS" \
        --output_path "$OUTPUT_DIR/judgments_${dim}.jsonl" \
        --scores_output_path "$OUTPUT_DIR/scores_${dim}.txt"
    echo "Scores for $dim:"
    cat "$OUTPUT_DIR/scores_${dim}.txt"
    echo ""
done

echo "=== Done. Results in $OUTPUT_DIR/ and $BASELINE_OUTPUT_DIR/ ==="
