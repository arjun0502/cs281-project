from mindeval.prompts import INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE

from pipeline import config
from pipeline.metric import run_judge
from pipeline.simulate import simulate_conversation


def regression_check(
    optimized_prompt: str,
    held_out_set: list[dict],
    dimension_just_optimized: str,
) -> dict[str, float]:
    """
    Simulate up to 20 held-out conversations with the optimized prompt,
    score all 5 dimensions, print a summary table, and return mean scores
    keyed by the full dimension score name.
    """
    patient_template = INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE.template
    all_scores: dict[str, list[float]] = {k: [] for k in config.DIMENSION_MAP.values()}

    for entry in held_out_set[:20]:
        member_details = entry["additional_context"]
        conversation = simulate_conversation(
            optimized_prompt, patient_template, member_details, config.N_TURNS
        )
        scores, _ = run_judge(conversation, member_details)
        for key in config.DIMENSION_MAP.values():
            all_scores[key].append(scores.get(key, 3.0))

    means = {k: sum(v) / len(v) for k, v in all_scores.items() if v}

    optimized_key = config.DIMENSION_MAP[dimension_just_optimized]
    print(f"\n{'Dimension':<45} {'Mean':>6}")
    print("-" * 55)
    for key, mean in means.items():
        marker = "  ← optimized" if key == optimized_key else ""
        print(f"{key:<45} {mean:>6.3f}{marker}")

    return means
