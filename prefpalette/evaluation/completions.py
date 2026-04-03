import logging
import os
import requests
import time
from collections import defaultdict
import random

__all__ = ["vllm_endpoint_chat_completions", "openai_chat_completions"]

CHAT_ENDPOINT = "/v1/chat/completions"


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start


def vllm_endpoint_chat_completions(
    messages_batch, model_name="test", max_new_tokens=None,
    temperature=1.0, request_timeout=10, max_retry=3,
    model_endpoint="http://localhost:8000", **kwargs,
):
    n_examples = len(messages_batch)
    responses = []
    if isinstance(messages_batch[0], dict):
        messages_batch = [messages_batch]
    usage_total = defaultdict(int)

    with Timer() as t:
        for messages in messages_batch:
            status_ok = False
            request_data = {"model": model_name, "messages": messages, "temperature": temperature}
            if max_new_tokens is not None:
                request_data["max_tokens"] = max_new_tokens
            headers = {"Content-Type": "application/json"}

            for _ in range(max_retry):
                try:
                    http_response = requests.post(
                        f"{model_endpoint}{CHAT_ENDPOINT}",
                        headers=headers, json=request_data, timeout=request_timeout,
                    )
                except Exception:
                    logging.error(f"Request failed, retrying after {request_timeout}s...")
                    time.sleep(request_timeout)
                    continue

                if http_response.status_code == 200:
                    response = http_response.json()
                    response_text = response["choices"][0]["message"]["content"]
                    for k, v in response["usage"].items():
                        usage_total[k] += v if v is not None else 0
                    if response_text and response_text[-1].lower() == "m":
                        status_ok = True
                        responses.append(response_text)
                        break
                logging.error(f"Bad response, retrying...")
                time.sleep(request_timeout)

            if not status_ok:
                responses.append(None)

    avg_time = [t.duration / n_examples] * len(responses)
    return dict(completions=responses, time_per_example=avg_time, usage_total=dict(usage_total))


def openai_chat_completions(
    messages_batch, model_name="gpt-4o", max_new_tokens=None,
    temperature=1.0, request_timeout=10, max_retry=5,
    api_account="openai", **kwargs,
):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    n_examples = len(messages_batch)
    responses = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with Timer() as t:
        for messages in messages_batch:
            for retry in range(max_retry):
                try:
                    response = client.chat.completions.create(
                        model=model_name, messages=messages,
                        temperature=temperature, max_tokens=max_new_tokens, timeout=30,
                    )
                    response_text = response.choices[0].message.content.strip()
                    usage = response.usage
                    usage_total["prompt_tokens"] += usage.prompt_tokens
                    usage_total["completion_tokens"] += usage.completion_tokens
                    usage_total["total_tokens"] += usage.total_tokens
                    responses.append(response_text)
                    break
                except Exception as e:
                    if retry == max_retry - 1:
                        logging.error(f"Failed after {max_retry} retries: {e}")
                        responses.append(None)
                    else:
                        time.sleep(random.randint(1, 10))

    avg_time = [t.duration / n_examples] * len(responses)
    return dict(completions=responses, time_per_example=avg_time, usage_total=usage_total)
