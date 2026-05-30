import json
import os

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
from gepa.utils import NoImprovementStopper

from mindeval.prompts import MINDEVAL_CLINICIAN_TEMPLATE

from prompt_opt_pipeline import config
from prompt_opt_pipeline.data_prep import build_emotional_escalation_splits
from prompt_opt_pipeline.metric import make_composite_metric
from prompt_opt_pipeline.utils import log, save_prompt, setup_logging


def main() -> None:
    setup_logging()
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.GEPA_RUNS_DIR, exist_ok=True)

    trainset, valset = build_emotional_escalation_splits()

    log(f"Trainset size: {len(trainset)} (hardest 25 emotional_escalation examples)")
    log(f"Valset size:   {len(valset)} (remaining 25 emotional_escalation examples)")
    log(f"Optimizing: composite metric (avg of all 5 dimensions)")

    # Baseline scores from existing v0_3 judge results (raw 1-6 scale)
    baseline_scores = [entry["baseline_score"] for entry in trainset]
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    log(f"Seed prompt mean score on trainset: {baseline_mean:.4f} (1-6 scale)")

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=config.MAX_METRIC_CALLS,
            run_dir=f"{config.GEPA_RUNS_DIR}/cycle_0/{config.OPTIMIZE_DIMENSION}",
            display_progress_bar=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=config.REFLECTION_LM,
        ),
        stop_callbacks=[
            NoImprovementStopper(max_iterations_without_improvement=config.MAX_ITERS_WITHOUT_IMPROVEMENT),
        ],
    )

    result = optimize_anything(
        seed_candidate=MINDEVAL_CLINICIAN_TEMPLATE.template,
        evaluator=make_composite_metric(),
        dataset=trainset,
        valset=valset,
        config=gepa_config,
    )

    seed_prompt = MINDEVAL_CLINICIAN_TEMPLATE.template
    prompt_changed = result.best_candidate.strip() != seed_prompt.strip()

    best_score_normalized = getattr(result, "best_score", None)
    best_score_raw = (best_score_normalized * 5.0 + 1.0) if best_score_normalized is not None else None

    summary = {
        "dimension": "composite",
        "dimension_label": "avg of all 5 dimensions",
        "trainset_size": len(trainset),
        "trainset_sizes_by_type": config.TRAINSET_SIZES_BY_TYPE,
        "max_metric_calls": config.MAX_METRIC_CALLS,
        "seed_mean_score_raw": round(baseline_mean, 4),
        "best_score_normalized": round(best_score_normalized, 4) if best_score_normalized is not None else None,
        "best_score_raw": round(best_score_raw, 4) if best_score_raw is not None else None,
        "per_example_seed_scores": baseline_scores,
        "prompt_changed": prompt_changed,
    }

    scores_path = f"{config.OUTPUTS_DIR}/optimization_scores.json"
    with open(scores_path, "w") as f:
        json.dump(summary, f, indent=2)

    save_prompt(result.best_candidate, f"{config.OUTPUTS_DIR}/optimized_prompt.txt")
    save_prompt(seed_prompt, f"{config.OUTPUTS_DIR}/seed_prompt.txt")

    log(f"\n{'='*60}")
    log(f"RESULTS")
    log(f"{'='*60}")
    log(f"Prompt changed:       {'YES' if prompt_changed else 'NO — same as seed'}")
    log(f"Seed mean score:      {baseline_mean:.4f} (1-6 scale)")
    if best_score_raw is not None:
        log(f"Best optimized score: {best_score_raw:.4f} (1-6 scale)")
        log(f"Delta:                {best_score_raw - baseline_mean:+.4f}")
    log(f"Optimized prompt → {config.OUTPUTS_DIR}/optimized_prompt.txt")
    log(f"Seed prompt       → {config.OUTPUTS_DIR}/seed_prompt.txt")
    log(f"Scores summary    → {scores_path}")


if __name__ == "__main__":
    main()
