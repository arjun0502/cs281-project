from string import Template

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mindeval.inference import InferenceEngine
from mindeval.judge_prompts import JUDGE_PROMPT_TEMPLATE_VERSION_DICT
from mindeval.prompts import INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE
from mindeval.utils import parse_judge_scores

from pipeline import config
from pipeline.simulate import simulate_conversation


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
)
def run_judge(conversation_str: str, member_details: dict) -> tuple[dict, str]:
    """
    Call the judge (gpt-4o, v0_3 template) on a conversation and parse all 5 scores.

    Returns (scores_dict, raw_judge_output).
    """
    judge_template = JUDGE_PROMPT_TEMPLATE_VERSION_DICT[config.JUDGE_TEMPLATE_VERSION]
    user_prompt = judge_template.safe_substitute(
        conversation_str=conversation_str,
        **member_details,
    )
    judge_engine = InferenceEngine(
        {"model": config.JUDGE_MODEL, "max_completion_tokens": 4096}
    )
    response, _ = judge_engine.generate_with_thinking(
        [{"role": "user", "content": user_prompt}]
    )
    scores = parse_judge_scores(response)[0]
    return scores, response


def make_metric(dimension: str):
    """
    Return a GEPA-compatible evaluator for the given dimension key.

    The returned function has signature:
        evaluator(candidate: str, example: dict) -> (float, dict)

    candidate: the candidate clinician prompt string being tested
    example:   dict with keys input, additional_context, baseline_score
    """
    dim_score_key = config.DIMENSION_MAP[dimension]
    patient_template = INTERACTION_MEMBER_ADVERSARIAL_TEMPLATE.template

    def evaluator(candidate: str, example: dict) -> tuple[float, dict]:
        member_details = example["additional_context"]

        conversation = simulate_conversation(
            candidate, patient_template, member_details, config.N_TURNS
        )
        scores, judge_output = run_judge(conversation, member_details)

        raw = scores.get(dim_score_key, 3.0)
        normalized = (raw - 1) / 5.0  # map [1, 6] → [0, 1]
        return normalized, {"feedback": judge_output}

    return evaluator
