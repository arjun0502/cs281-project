"""
Minimal integration test — uses real API calls but at the smallest possible scale.
Runs 1 cycle, 1 dimension (therapeutic_alliance), 3 metric calls, 2-example trainset.

Purpose: verify end-to-end wiring and estimate cost before the full run.
Check your OpenAI dashboard balance before and after, then multiply by the scale factor printed below.

Run from project root:
    python pipeline/tests/integration.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import prompt_opt_pipeline.config as config
config.MAX_METRIC_CALLS = 10
config.MAX_ITERS_WITHOUT_IMPROVEMENT = 2
config.TRAINSET_SIZE = 2
config.REGRESSION_SET_MAX = 2
config.MAX_CYCLES = 1
config.N_TURNS = 10
config.DIMENSION_ORDER = ["therapeutic_alliance"]

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
from gepa import NoImprovementStopper
from mindeval.prompts import MINDEVAL_CLINICIAN_TEMPLATE
from prompt_opt_pipeline.data_prep import build_regression_set, build_trainsets, load_all_pairs
from prompt_opt_pipeline.metric import make_metric
from prompt_opt_pipeline.regression import regression_check
from prompt_opt_pipeline.utils import save_prompt, setup_logging


def main():
    setup_logging()
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.GEPA_RUNS_DIR, exist_ok=True)

    scale_factor = (10 / config.N_TURNS) * (50 / config.MAX_METRIC_CALLS) * 5 * 3
    print("=== INTEGRATION TEST ===")
    print(f"  1 dimension, {config.N_TURNS} turns, {config.MAX_METRIC_CALLS} metric calls, 1 cycle")
    print(f"  Scale factor to full run: {scale_factor:.0f}x")
    print(f"  Check OpenAI dashboard before + after, multiply difference by {scale_factor:.0f}x")
    print()

    t0 = time.time()

    all_pairs = load_all_pairs()
    trainsets = build_trainsets(all_pairs)
    held_out_set = build_regression_set(all_pairs, trainsets)
    print(f"[OK] data loaded")

    dimension = "therapeutic_alliance"
    gepa_config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=config.MAX_METRIC_CALLS,
            run_dir=f"{config.GEPA_RUNS_DIR}/integration_test/{dimension}",
            display_progress_bar=True,
        ),
        reflection=ReflectionConfig(reflection_lm=config.REFLECTION_LM),
        stop_callbacks=NoImprovementStopper(
            max_iterations_without_improvement=config.MAX_ITERS_WITHOUT_IMPROVEMENT
        ),
    )

    print(f"[...] GEPA optimizing {dimension}...")
    t1 = time.time()
    result = optimize_anything(
        seed_candidate=MINDEVAL_CLINICIAN_TEMPLATE.template,
        evaluator=make_metric(dimension),
        dataset=trainsets[dimension],
        config=gepa_config,
    )
    print(f"[OK] GEPA done in {time.time() - t1:.1f}s")

    print(f"[...] Regression check...")
    t2 = time.time()
    regression_check(result.best_candidate, held_out_set, dimension)
    print(f"[OK] Regression check done in {time.time() - t2:.1f}s")

    print(f"\n=== DONE — total time: {time.time() - t0:.1f}s ===")
    save_prompt(result.best_candidate, f"{config.OUTPUTS_DIR}/integration_test_prompt.txt")
    print(f"Best prompt saved to {config.OUTPUTS_DIR}/integration_test_prompt.txt")


if __name__ == "__main__":
    main()
