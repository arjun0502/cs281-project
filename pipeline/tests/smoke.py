"""
Smoke tests for the GEPA pipeline.

Patches InferenceEngine.generate_with_thinking to return canned responses
so no real API calls (and no OpenAI credits) are consumed.

Run from the project root:
    python gepa/tests/smoke.py
"""

import os
import sys

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# InferenceEngine calls OpenAI(max_retries=5) in __init__ which checks for an API key.
# Set a dummy key so the client initialises without error — no real calls are made
# because generate_with_thinking is patched below.
os.environ.setdefault("OPENAI_API_KEY", "sk-smoke-test-dummy-key")

from unittest.mock import patch

FAKE_CLINICIAN  = "I hear you. How has your week been?"
FAKE_MEMBER     = "Not great, honestly."
FAKE_JUDGE_OUT  = (
    "Clinical Accuracy & Competence: 4\n"
    "Ethical & Professional Conduct: 4\n"
    "Assessment & Response: 4\n"
    "Therapeutic Relationship & Alliance: 4\n"
    "AI-Specific Communication Quality: 4\n"
)


def _fake_generate(self, messages):
    # Route by content: if last message references evaluation criteria → judge call
    last_content = messages[-1].get("content", "") if messages else ""
    if "Clinical Accuracy" in last_content:
        return FAKE_JUDGE_OUT, ""
    # Distinguish member from clinician by system prompt keyword
    sys_content = messages[0].get("content", "") if messages else ""
    if "client" in sys_content.lower() or "roleplay" in sys_content.lower():
        return FAKE_MEMBER, ""
    return FAKE_CLINICIAN, ""


def run_smoke_tests():
    with patch("mindeval.inference.InferenceEngine.generate_with_thinking", _fake_generate):

        # ── 1. Data loading ──────────────────────────────────────────────
        from pipeline.data_prep import build_regression_set, build_trainsets, load_all_pairs

        pairs = load_all_pairs()
        assert len(pairs) == 200, f"Expected 200 pairs, got {len(pairs)}"

        trainsets = build_trainsets(pairs)
        from pipeline import config
        assert set(trainsets.keys()) == set(config.DIMENSION_ORDER), "Missing dimension keys"
        for dim, entries in trainsets.items():
            assert len(entries) == config.TRAINSET_SIZE, (
                f"{dim}: expected {config.TRAINSET_SIZE} entries, got {len(entries)}"
            )

        held_out = build_regression_set(pairs, trainsets)
        assert 0 < len(held_out) <= config.REGRESSION_SET_MAX, (
            f"Regression set size {len(held_out)} out of range"
        )
        print(
            f"[OK] data_prep — 200 pairs, trainsets={config.TRAINSET_SIZE}/dim, "
            f"held_out={len(held_out)}"
        )

        # Verify trainset entries have required fields
        sample = trainsets["therapeutic_alliance"][0]
        assert "input" in sample
        assert "additional_context" in sample
        assert "baseline_score" in sample
        print("[OK] data_prep — trainset entry schema correct")

        # ── 2. Simulation ────────────────────────────────────────────────
        from mindeval.prompts import (
            INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE,
            MINDEVAL_CLINICIAN_TEMPLATE,
        )
        from pipeline.simulate import simulate_conversation

        member = pairs[0]["member_profile"]
        convo = simulate_conversation(
            MINDEVAL_CLINICIAN_TEMPLATE.template,
            INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE.template,
            member,
            num_turns=2,
        )
        assert "<member>" in convo, "Missing <member> tags"
        assert "<therapist>" in convo, "Missing <therapist> tags"
        # 2 turns → 2 clinician + 2 member + initial Hello = 5 messages, but
        # messages_to_convo_str includes the initial Hello user message too
        turns = convo.count("<therapist>")
        assert turns == 2, f"Expected 2 clinician turns, got {turns}"
        print(f"[OK] simulate_conversation — {turns} clinician turns, format correct")

        # ── 3. Judge ─────────────────────────────────────────────────────
        from pipeline.metric import run_judge

        scores, raw = run_judge(convo, member)
        required_keys = set(config.DIMENSION_MAP.values())
        assert required_keys.issubset(scores.keys()), (
            f"Missing score keys: {required_keys - set(scores.keys())}"
        )
        print(f"[OK] run_judge — all 5 dimension keys present, sample score={scores['Therapeutic Relationship & Alliance']}")

        # ── 4. Metric ────────────────────────────────────────────────────
        from pipeline.metric import make_metric

        metric_fn = make_metric("therapeutic_alliance")
        example = trainsets["therapeutic_alliance"][0]
        score, meta = metric_fn(MINDEVAL_CLINICIAN_TEMPLATE.template, example)
        assert 0.0 <= score <= 1.0, f"Normalized score out of [0,1]: {score}"
        assert "feedback" in meta, "Missing 'feedback' key in metric return"
        print(f"[OK] make_metric — score={score:.3f}, feedback present")

        # ── 5. Regression check ──────────────────────────────────────────
        from pipeline.regression import regression_check

        means = regression_check(
            MINDEVAL_CLINICIAN_TEMPLATE.template,
            held_out[:2],
            "therapeutic_alliance",
        )
        assert isinstance(means, dict), "regression_check must return a dict"
        assert required_keys.issubset(means.keys()), "Missing dimension keys in means"
        print(f"[OK] regression_check — returned means for all 5 dimensions")

    print("\n✓ All smoke tests passed — no API credits consumed.")


if __name__ == "__main__":
    run_smoke_tests()
