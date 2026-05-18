from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from mindeval.utils import separate_thinking_from_response


class InferenceEngine:
    def __init__(self, api_params: dict[str, str]):
        self.api_params = api_params
        self.model = api_params["model"]
        self.max_completion_tokens = api_params.get("max_completion_tokens", 4096)
        self.max_workers = api_params.get("max_workers", 10)
        self.client = OpenAI(max_retries=5)

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response.choices[0].message.content

    def batch_generate(self, list_of_messages: list[list[dict[str, str]]]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.generate, messages) for messages in list_of_messages]
            return [f.result() for f in futures]

    def generate_with_thinking(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        response = None
        while response is None:
            response = self.generate(messages)
            if response is None:
                print("Retrying generation because response is None...")
        parts = separate_thinking_from_response(response)
        return parts["response"], parts["thinking"]

    def batch_generate_with_thinking(
        self, list_of_messages: list[list[dict[str, str]]]
    ) -> tuple[list[str], list[str]]:
        responses = self.batch_generate(list_of_messages)
        texts = []
        thinking_traces = []
        for response in responses:
            parts = separate_thinking_from_response(response)
            texts.append(parts["response"])
            thinking_traces.append(parts["thinking"])
        return texts, thinking_traces
