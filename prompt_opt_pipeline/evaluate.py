"""
Post-optimization evaluation.

Evaluates the optimized prompt on:
  - 25 held-out emotional_escalation examples (valset)
  - 25 randomly sampled examples from each of the other 3 adversarial types

Saves interactions and judgments to results/adversarial/gpt4omini_clinician_optimized_prompt/.

Usage:
    python -m prompt_opt_pipeline.evaluate
"""

import json
import os
import random
import shutil
import statistics

from mindeval.utils import load_jsonl
from mindeval.prompts import INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE

from prompt_opt_pipeline import config
from prompt_opt_pipeline.data_prep import build_emotional_escalation_splits, load_pairs_by_type
from prompt_opt_pipeline.metric import run_judge
from prompt_opt_pipeline.simulate import simulate_conversation

EVAL_N = 25
RANDOM_SEED = 42
RESULTS_DIR = "results/adversarial/gpt4omini_clinician_optimized_prompt"


def simulate_and_score(
    optimized_prompt: str, examples: list[dict]
) -> tuple[dict[str, list[float]], list[dict]]:
    """
    Run optimized prompt on examples.
    Returns ({dim_label: [scores]}, [interaction records]).
    """
    patient_template = INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE.template
    dim_totals: dict[str, list[float]] = {k: [] for k in config.DIMENSION_MAP.values()}
    records = []

    for i, example in enumerate(examples):
        member_details = example["additional_context"]
        conversation = simulate_conversation(
            optimized_prompt, patient_template, member_details, config.N_TURNS
        )
        scores, judge_output = run_judge(conversation, member_details)
        for label in config.DIMENSION_MAP.values():
            score = scores.get(label)
            if score is not None:
                dim_totals[label].append(score)
        records.append({
            "member_profile":    member_details,
            "interaction_str":   conversation,
            "parsed_judgment":   scores,
            "raw_judge_output":  judge_output,
        })
        print(f"    [{i+1}/{len(examples)}] done", flush=True)

    return dim_totals, records


def seed_means_from_entries(examples: list[dict]) -> dict[str, float]:
    return {
        label: statistics.mean([ex["baseline_dim_scores"].get(label, 3.0) for ex in examples])
        for label in config.DIMENSION_MAP.values()
    }


def seed_means_from_judgments(adv_type: str, indices: list[int]) -> dict[str, float]:
    judgs = load_jsonl(f"{config.JUDGMENTS_DIR}/judgments_{adv_type}.jsonl")
    selected = [judgs[i] for i in indices]
    return {
        label: statistics.mean([j["parsed_judgment"].get(label, 3.0) for j in selected])
        for label in config.DIMENSION_MAP.values()
    }


def save_interactions(adv_type: str, records: list[dict]) -> None:
    interactions_dir = f"{RESULTS_DIR}/interactions"
    os.makedirs(interactions_dir, exist_ok=True)
    path = f"{interactions_dir}/{adv_type}.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"    Interactions saved → {path}")


def print_table(results: dict) -> None:
    dim_labels = list(config.DIMENSION_MAP.values())
    col_w = 34
    header = f"{'Dimension':<{col_w}} {'Seed':>6} {'Opt':>6} {'Delta':>7}"
    sep = "=" * len(header)

    for adv_type, data in results.items():
        print(f"\nAdversarial type: {adv_type} (n={data['n']})")
        print(sep)
        print(header)
        print(sep)
        for label in dim_labels:
            s = data["seed"].get(label, float("nan"))
            o = data["optimized"].get(label, float("nan"))
            delta = o - s
            print(f"{label:<{col_w}} {s:>6.2f} {o:>6.2f} {delta:>+7.2f}")
        print(sep)


def main() -> None:
    prompt_path = f"{config.OUTPUTS_DIR}/optimized_prompt.txt"
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Optimized prompt not found at {prompt_path}. "
            "Run python -m prompt_opt_pipeline.main first."
        )

    with open(prompt_path) as f:
        optimized_prompt = f.read()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    by_type = load_pairs_by_type()
    _, valset = build_emotional_escalation_splits()

    all_results = {}

    # emotional_escalation — use held-out valset
    print(f"\nEvaluating emotional_escalation valset ({len(valset)} examples)...")
    opt_scores, records = simulate_and_score(optimized_prompt, valset)
    save_interactions("emotional_escalation", records)
    all_results["emotional_escalation"] = {
        "n": len(valset),
        "seed": seed_means_from_entries(valset),
        "optimized": {label: statistics.mean(v) if v else float("nan") for label, v in opt_scores.items()},
    }

    # other adversarial types — random sample of 25
    for adv_type in config.ADVERSARIAL_TYPES:
        if adv_type == "emotional_escalation":
            continue
        pairs = by_type[adv_type]
        indices = rng.sample(range(len(pairs)), min(EVAL_N, len(pairs)))
        examples = [
            {
                "additional_context":  pairs[i]["member_profile"],
                "baseline_dim_scores": {label: pairs[i]["scores"].get(label, 3.0) for label in config.DIMENSION_MAP.values()},
            }
            for i in indices
        ]
        print(f"\nEvaluating {adv_type} ({len(examples)} random examples)...")
        opt_scores, records = simulate_and_score(optimized_prompt, examples)
        save_interactions(adv_type, records)
        all_results[adv_type] = {
            "n": len(examples),
            "seed": seed_means_from_judgments(adv_type, indices),
            "optimized": {label: statistics.mean(v) if v else float("nan") for label, v in opt_scores.items()},
        }

    print_table(all_results)

    results_path = f"{RESULTS_DIR}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    shutil.copy(f"{config.OUTPUTS_DIR}/optimized_prompt.txt", f"{RESULTS_DIR}/optimized_prompt.txt")

    print(f"\nResults saved  → {results_path}")
    print(f"Prompt saved   → {RESULTS_DIR}/optimized_prompt.txt")
    print(f"Interactions   → {RESULTS_DIR}/interactions/")


if __name__ == "__main__":
    main()
