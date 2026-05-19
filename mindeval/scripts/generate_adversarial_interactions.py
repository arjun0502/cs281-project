from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from jsonargparse import CLI
from tqdm import tqdm

from mindeval.inference import InferenceEngine
from mindeval.prompts import (
    ADVERSARIAL_MEMBER_INSTRUCTIONS,
    INTERACTION_CLINICIAN_VERSION_DICT,
    INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE,
)
from mindeval.scripts.generate_interactions import run_interactions
from mindeval.utils import load_jsonl, save_interactions_to_text, save_to_jsonl

ADVERSARIAL_TYPES = list(ADVERSARIAL_MEMBER_INSTRUCTIONS.keys())


def main(
    profiles_path: str,
    clinician_system_template_version: str,
    clinician_model_api_params: dict[str, Any],
    member_model_api_params: dict[str, Any],
    n_turns: int,
    max_workers: int,
    output_dir: str,
    adversarial_type: str = None,
):
    """
    Run adversarial interactions for one or all adversarial dimensions.

    Outputs one JSONL and one text file per adversarial type inside output_dir:
        {output_dir}/{adversarial_type}.jsonl
        {output_dir}/{adversarial_type}.txt

    If adversarial_type is omitted, all four dimensions are run.
    The member template is always v0_2_adversarial; clinician template is configurable.
    """
    if adversarial_type is not None and adversarial_type not in ADVERSARIAL_TYPES:
        raise ValueError(
            f"Unknown adversarial_type {adversarial_type!r}. "
            f"Must be one of: {ADVERSARIAL_TYPES}"
        )
    types_to_run = [adversarial_type] if adversarial_type else ADVERSARIAL_TYPES

    # load profiles
    raw_profiles = load_jsonl(profiles_path)
    base_profiles = []
    for p in raw_profiles:
        d = p["member_attributes"]
        d["member_narrative"] = p["member_narrative"]
        base_profiles.append(d)

    clinician_template = INTERACTION_CLINICIAN_VERSION_DICT[clinician_system_template_version]
    clinician_model = InferenceEngine(clinician_model_api_params)
    member_model = InferenceEngine(member_model_api_params)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for adv_type in types_to_run:
        instructions = ADVERSARIAL_MEMBER_INSTRUCTIONS[adv_type]
        profiles = [
            {**p, "adversarial_instructions": instructions} for p in base_profiles
        ]

        def process_profile(idx_profile):
            idx, profile = idx_profile
            interaction, clinician_thinking_traces, member_thinking_traces = run_interactions(
                profile,
                clinician_template,
                clinician_model,
                INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE,
                member_model,
                n_turns,
            )
            return idx, {
                "member_profile": profile,
                "adversarial_type": adv_type,
                "interaction": interaction,
                "clinician_thinking_traces": clinician_thinking_traces,
                "member_thinking_traces": member_thinking_traces,
            }

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_profile, (idx, profile)): idx
                for idx, profile in enumerate(profiles)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(profiles),
                desc=f"[{adv_type}]",
            ):
                idx, result = future.result()
                results[idx] = result

        all_interactions = [results[i] for i in sorted(results)]
        save_to_jsonl(all_interactions, str(out / f"{adv_type}.jsonl"))
        save_interactions_to_text(all_interactions, str(out / f"{adv_type}.txt"))
        print(f"Saved {len(all_interactions)} interactions → {out / adv_type}")


if __name__ == "__main__":
    CLI([main], as_positional=False)
