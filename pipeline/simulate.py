from string import Template

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mindeval.inference import InferenceEngine
from mindeval.utils import messages_to_convo_str
from pipeline import config


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
)
def _generate(engine: InferenceEngine, messages: list[dict]) -> tuple[str, str]:
    return engine.generate_with_thinking(messages)


def simulate_conversation(
    clinician_system_prompt: str,
    patient_system_prompt: str,
    member_details: dict,
    num_turns: int,
) -> str:
    """
    Substitute member_details into both prompt templates (safe_substitute),
    then run num_turns exchanges — member speaks first via the initial 'Hello'
    message, clinician responds, alternating until num_turns complete.

    Returns the full conversation as an XML-tagged string via messages_to_convo_str.
    """
    clinician_sys = Template(clinician_system_prompt).safe_substitute(**member_details)
    patient_sys   = Template(patient_system_prompt).safe_substitute(**member_details)

    clinician_engine = InferenceEngine(
        {"model": config.CLINICIAN_MODEL, "max_completion_tokens": 4096}
    )
    member_engine = InferenceEngine(
        {"model": config.MEMBER_MODEL, "max_completion_tokens": 4096}
    )

    # Mirror run_interactions from generate_interactions.py exactly
    clinician_messages: list[dict] = [
        {"role": "system",    "content": clinician_sys},
        {"role": "user",      "content": "Hello"},
    ]
    member_messages: list[dict] = [
        {"role": "system",    "content": patient_sys},
        {"role": "assistant", "content": "Hello"},
    ]

    for _ in range(num_turns):
        clinician_response, _ = _generate(clinician_engine, clinician_messages)
        clinician_response = clinician_response or "..."
        clinician_messages.append({"role": "assistant", "content": clinician_response})
        member_messages.append({"role": "user",      "content": clinician_response})

        member_response, _ = _generate(member_engine, member_messages)
        member_response = member_response or "..."
        member_messages.append({"role": "assistant", "content": member_response})
        clinician_messages.append({"role": "user",   "content": member_response})

    # clinician_messages[1:] drops the system prompt; keeps 'Hello' onward
    return messages_to_convo_str(clinician_messages[1:])
