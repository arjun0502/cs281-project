#!/usr/bin/env bash
set -euo pipefail

CLINICIAN_MODEL="${1:-gpt4omini}"
JUDGE_VERSION="${2:-v0_3}"
JUDGE_MODEL="${3:-gpt-4o}"
MAX_WORKERS="${4:-50}"

INTERACTIONS_DIR="results/adversarial/${CLINICIAN_MODEL}_clinician/interactions"
OUTPUT_DIR="results/adversarial/${CLINICIAN_MODEL}_clinician/${JUDGE_VERSION}_judge"
BASELINE_INTERACTIONS="results/baseline/${CLINICIAN_MODEL}_clinician/interactions/interactions_baseline.jsonl"
BASELINE_OUTPUT_DIR="results/baseline/${CLINICIAN_MODEL}_clinician/${JUDGE_VERSION}_judge"

ADVERSARIAL_TYPES=(pushback validation_seeking emotional_escalation dependency)

echo "=== Judging baseline with ${JUDGE_VERSION} judge ==="
mkdir -p "$BASELINE_OUTPUT_DIR"

python mindeval/scripts/generate_judgments.py \
    --interactions_path "$BASELINE_INTERACTIONS" \
    --judge_template_version "$JUDGE_VERSION" \
    --judge_model_api_params "{'model':'$JUDGE_MODEL','max_completion_tokens':4096}" \
    --max_workers "$MAX_WORKERS" \
    --output_path "$BASELINE_OUTPUT_DIR/judgments_baseline.jsonl" \
    --scores_output_path "$BASELINE_OUTPUT_DIR/scores_baseline.txt"

echo "Baseline scores:"
cat "$BASELINE_OUTPUT_DIR/scores_baseline.txt"
echo ""

echo "=== Re-judging adversarial interactions with ${JUDGE_VERSION} judge ==="
echo "Interactions: $INTERACTIONS_DIR | Output: $OUTPUT_DIR"

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

echo "=== Done. Results in $OUTPUT_DIR/ and $BASELINE_OUTPUT_DIR/ ==="
