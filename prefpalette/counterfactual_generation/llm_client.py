import requests
import time
import logging

CHAT_ENDPOINT = "/v1/chat/completions"
COMPLETIONS_ENDPOINT = "/v1/completions"


class VLLMClient:
    """Client for interacting with a vLLM-compatible API endpoint."""

    def __init__(self, model_endpoint, model_name):
        self.model_endpoint = model_endpoint
        self.model_name = model_name

    def chat_completion(self, messages, model_name=None, max_tokens=512,
                        temperature=1.0, request_timeout=120, max_retry=3):
        model_name = model_name or self.model_name
        request_data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}

        for retry in range(max_retry):
            try:
                http_response = requests.post(
                    f"{self.model_endpoint}{CHAT_ENDPOINT}",
                    headers=headers,
                    json=request_data,
                    timeout=request_timeout,
                )
                if http_response.status_code == 200:
                    response = http_response.json()
                    return {
                        "status_ok": True,
                        "content": response["choices"][0]["message"]["content"],
                        "usage": response["usage"],
                    }
                logging.warning(f"Request failed (status {http_response.status_code}): {http_response.text}")
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request error: {e}")
            time.sleep(request_timeout)

        logging.error(f"Max retries ({max_retry}) reached")
        return {"status_ok": False, "content": None, "usage": None}
