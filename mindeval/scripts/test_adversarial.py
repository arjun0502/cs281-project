"""
Test script comparing two approaches for adversarial patient simulation.

Approach 1: Adversarial instructions injected into the member system prompt
            (INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE). Profile unchanged.

Approach 2: Fixed demographic/circumstantial attributes kept as-is. Behavioral
            attributes (thought_process, general_outlook, conversation_style,
            recent_mood, attitude_towards_mindfulness) and member_narrative
            regenerated from scratch by the LLM, shaped by the adversarial pattern.
            Base member template (INTERACTION_MEMBER_TEMPLATE) unchanged.

Runs on the first N_PROFILES profiles from profiles_test.jsonl.
Outputs to results/adversarial_test/{name}/approach{1,2}_{type}/
"""

import json
import re
from pathlib import Path

from mindeval.inference import InferenceEngine
from mindeval.prompts import (
    ADVERSARIAL_MEMBER_INSTRUCTIONS,
    ADVERSARIAL_PROFILE_DESCRIPTIONS,
    ADVERSARIAL_PROFILE_GENERATION_TEMPLATE,
    INTERACTION_CLINICIAN_VERSION_DICT,
    INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE,
    INTERACTION_MEMBER_VERSION_DICT,
)
from mindeval.scripts.generate_interactions import run_interactions
from mindeval.utils import load_jsonl, save_to_jsonl

N_PROFILES = 3
N_TURNS = 10
ADVERSARIAL_TYPES = ["pushback", "validation_seeking", "emotional_escalation", "dependency"]

# Demographic/circumstantial attributes that are never changed in Approach 2.
FIXED_ATTRS = {
    "name", "sex", "gender_identity", "sexual_orientation", "age", "race",
    "education", "profession", "employment_status", "financial_situation",
    "siblings", "relationship_status", "living_situation", "exercise",
    "sleep_quality", "region", "depressive_symptoms", "anxious_symptoms",
    "program_goal", "support_system",
}

# Behavioral attributes that Approach 2 regenerates from scratch.
BEHAVIORAL_ATTRS = {
    "thought_process", "general_outlook", "conversation_style",
    "recent_mood", "attitude_towards_mindfulness", "member_narrative",
}

# Keys that go into member_attributes in the saved profile (excludes member_narrative).
MEMBER_ATTRIBUTE_KEYS = list(FIXED_ATTRS | (BEHAVIORAL_ATTRS - {"member_narrative"}))


def generate_adversarial_profile(
    profile: dict, adversarial_type: str, model: InferenceEngine
) -> dict:
    """
    Generate new behavioral attributes + narrative for a patient, shaped by the
    adversarial pattern. Fixed demographic attributes are passed in as context
    but not changed. Returns a dict with the new behavioral fields only.
    """
    prompt = ADVERSARIAL_PROFILE_GENERATION_TEMPLATE.substitute(
        **profile,
        adversarial_type=adversarial_type,
        adversarial_description=ADVERSARIAL_PROFILE_DESCRIPTIONS[adversarial_type],
    )
    raw = model.generate([{"role": "user", "content": prompt}])

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    generated = json.loads(raw)

    # Validate expected keys are present
    missing = BEHAVIORAL_ATTRS - set(generated.keys())
    if missing:
        raise ValueError(f"LLM output missing expected fields: {missing}")

    return {k: generated[k] for k in BEHAVIORAL_ATTRS}


def save_run(result: list[dict], directory: Path, label: str):
    """Save interaction as JSONL and a human-readable text file."""
    directory.mkdir(parents=True, exist_ok=True)
    save_to_jsonl(result, str(directory / "interaction.jsonl"))
    with open(directory / "interaction.txt", "w") as f:
        f.write(f"{label}\n{'='*60}\n\n")
        for package in result:
            p = package["member_profile"]
            f.write(f"Profile: {p['name']}, age {p['age']}, {p['profession']}\n")
            f.write(f"Depressive: {p['depressive_symptoms']} | Anxious: {p['anxious_symptoms']}\n")
            f.write(f"Goal: {p['program_goal']}\n")
            f.write(f"{'='*60}\n\n")
            for turn in package["interaction"]:
                role = "Clinician" if turn["role"] == "assistant" else "Member"
                f.write(f"[{role}]\n{turn['content']}\n\n")


def main():
    output_dir = Path("results/adversarial_test")

    clinician_model = InferenceEngine({"model": "gpt-4o-mini", "max_completion_tokens": 4096})
    member_model = InferenceEngine({"model": "gpt-4o-mini", "max_completion_tokens": 4096})
    profile_gen_model = InferenceEngine({"model": "gpt-4o", "max_completion_tokens": 4096})

    clinician_template = INTERACTION_CLINICIAN_VERSION_DICT["v0_1"]
    base_member_template = INTERACTION_MEMBER_VERSION_DICT["v0_2"]

    raw_profiles = load_jsonl("data/profiles_test.jsonl")[:N_PROFILES]
    profiles = [
        {**p["member_attributes"], "member_narrative": p["member_narrative"]}
        for p in raw_profiles
    ]

    # Accumulate adversarial profiles per type to save as combined JSONL files.
    adversarial_profiles_by_type: dict[str, list[dict]] = {t: [] for t in ADVERSARIAL_TYPES}

    for profile in profiles:
        name = profile["name"]
        print(f"\nProfile: {name}")
        profile_dir = output_dir / name.lower()

        for adv_type in ADVERSARIAL_TYPES:
            print(f"  [{adv_type}]")

            # ------------------------------------------------------------------
            # Approach 1: adversarial instructions in the template, profile unchanged
            # ------------------------------------------------------------------
            print(f"    approach 1 ...", end=" ", flush=True)
            profile_a1 = {
                **profile,
                "adversarial_instructions": ADVERSARIAL_MEMBER_INSTRUCTIONS[adv_type],
            }
            interaction_a1, _, _ = run_interactions(
                profile_a1,
                clinician_template,
                clinician_model,
                INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE,
                member_model,
                N_TURNS,
            )
            save_run(
                [{"member_profile": profile_a1, "interaction": interaction_a1}],
                profile_dir / f"approach1_{adv_type}",
                label=f"APPROACH 1 — modified template | {adv_type}",
            )
            print("done")

            # ------------------------------------------------------------------
            # Approach 2: regenerate behavioral attributes + narrative, base template
            # ------------------------------------------------------------------
            print(f"    approach 2 (generating adversarial profile) ...", end=" ", flush=True)
            new_behavioral = generate_adversarial_profile(profile, adv_type, profile_gen_model)
            adversarial_profile = {**profile, **new_behavioral}

            interaction_a2, _, _ = run_interactions(
                adversarial_profile,
                clinician_template,
                clinician_model,
                base_member_template,
                member_model,
                N_TURNS,
            )
            a2_dir = profile_dir / f"approach2_{adv_type}"
            save_run(
                [{"member_profile": adversarial_profile, "interaction": interaction_a2}],
                a2_dir,
                label=f"APPROACH 2 — regenerated profile | {adv_type}",
            )

            # Save adversarial profile in profiles.jsonl format for later reuse.
            profile_record = {
                "member_attributes": {
                    k: adversarial_profile[k]
                    for k in MEMBER_ATTRIBUTE_KEYS
                    if k in adversarial_profile
                },
                "member_narrative": adversarial_profile["member_narrative"],
                "member_narrative_thinking_trace": "",
            }
            adversarial_profiles_by_type[adv_type].append(profile_record)
            print("done")

    # Save one combined profiles JSONL per adversarial type.
    output_dir.mkdir(parents=True, exist_ok=True)
    for adv_type, records in adversarial_profiles_by_type.items():
        out_path = output_dir / f"adversarial_profiles_{adv_type}.jsonl"
        save_to_jsonl(records, str(out_path))
        print(f"Saved {len(records)} profiles → {out_path}")

    print(f"\nAll done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
