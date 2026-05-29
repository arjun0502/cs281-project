import os

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
from gepa import NoImprovementStopper

from mindeval.prompts import MINDEVAL_CLINICIAN_TEMPLATE

from pipeline import config
from pipeline.data_prep import build_regression_set, build_trainsets, load_all_pairs
from pipeline.metric import make_metric
from pipeline.regression import regression_check
from pipeline.utils import log, save_prompt, setup_logging


def main() -> None:
    setup_logging()
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.GEPA_RUNS_DIR, exist_ok=True)

    all_pairs = load_all_pairs()
    trainsets = build_trainsets(all_pairs)
    held_out_set = build_regression_set(all_pairs, trainsets)

    log(f"Trainset sizes: { {d: len(t) for d, t in trainsets.items()} }")
    log(f"Held-out set size: {len(held_out_set)}")

    seed_prompt: str = MINDEVAL_CLINICIAN_TEMPLATE.template
    prev_cycle_means: dict | None = None

    for cycle in range(config.MAX_CYCLES):
        log(f"\n{'='*60}\nCYCLE {cycle + 1}\n{'='*60}")
        cycle_means: dict[str, dict] = {}

        for dimension in config.DIMENSION_ORDER:
            log(f"\n--- Optimizing: {dimension} ---")

            gepa_config = GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=config.MAX_METRIC_CALLS,
                    run_dir=f"{config.GEPA_RUNS_DIR}/cycle_{cycle}/{dimension}",
                    display_progress_bar=True,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=config.REFLECTION_LM,
                ),
                stop_callbacks=NoImprovementStopper(
                    max_iterations_without_improvement=config.MAX_ITERS_WITHOUT_IMPROVEMENT
                ),
            )

            result = optimize_anything(
                seed_candidate=seed_prompt,
                evaluator=make_metric(dimension),
                dataset=trainsets[dimension],
                config=gepa_config,
            )

            best_prompt: str = result.best_candidate

            means = regression_check(best_prompt, held_out_set, dimension)
            cycle_means[dimension] = means

            regressed = _check_regression(means, prev_cycle_means)
            if not regressed:
                seed_prompt = best_prompt
                log(f"[ACCEPTED] {dimension}")
            else:
                log(f"[WARNING] Regression detected for {dimension} — not advancing seed")
                save_prompt(
                    best_prompt,
                    f"{config.OUTPUTS_DIR}/rejected_{dimension}_cycle{cycle}.txt",
                )

            save_prompt(seed_prompt, f"{config.OUTPUTS_DIR}/best_prompt_{dimension}.txt")

        if prev_cycle_means is not None:
            avg_imp = _avg_improvement(cycle_means, prev_cycle_means)
            log(f"Average improvement this cycle: {avg_imp:.4f}")
            if avg_imp < config.PLATEAU_THRESHOLD:
                log("Plateau reached — stopping early")
                break

        prev_cycle_means = cycle_means

    save_prompt(seed_prompt, f"{config.OUTPUTS_DIR}/final_clinician_prompt.txt")
    log(f"\nDone. Final prompt → {config.OUTPUTS_DIR}/final_clinician_prompt.txt")


def _check_regression(
    current_means: dict[str, float],
    prev_cycle_means: dict | None,
) -> bool:
    if prev_cycle_means is None:
        return False
    for prev_means in prev_cycle_means.values():
        for key in config.DIMENSION_MAP.values():
            prev_score = prev_means.get(key, 0.0)
            curr_score = current_means.get(key, 0.0)
            if prev_score - curr_score > config.REGRESSION_THRESHOLD:
                return True
    return False


def _avg_improvement(
    current: dict[str, dict],
    previous: dict[str, dict],
) -> float:
    improvements = []
    for dim in config.DIMENSION_ORDER:
        if dim in current and dim in previous:
            for key in config.DIMENSION_MAP.values():
                improvements.append(
                    current[dim].get(key, 0.0) - previous[dim].get(key, 0.0)
                )
    return sum(improvements) / len(improvements) if improvements else 0.0


if __name__ == "__main__":
    main()
