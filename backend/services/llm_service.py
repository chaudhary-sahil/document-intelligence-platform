"""Talks to a local LLM through Ollama's HTTP API (no extra library needed)."""
import requests

from backend.config import LLM_TIMEOUT_SECONDS, OLLAMA_MODEL, OLLAMA_URL


class LLMError(RuntimeError):
    """Raised with a human-readable message when the LLM can't be used."""


def generate(system_prompt, user_prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Low temperature = factual, repeatable answers (we want grounding,
        # not creativity). num_ctx is how many tokens the model can read.
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT_SECONDS
        )
    except requests.ConnectionError as exc:
        raise LLMError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Install it from ollama.com, "
            "start it (open the app or run `ollama serve`), then try again."
        ) from exc
    except requests.Timeout as exc:
        raise LLMError(
            f"The model did not answer within {LLM_TIMEOUT_SECONDS}s. "
            "Try again, or use a smaller model."
        ) from exc

    if response.status_code == 404:
        raise LLMError(
            f"Model '{OLLAMA_MODEL}' is not installed. Run: ollama pull {OLLAMA_MODEL}"
        )
    if not response.ok:
        raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")

    return response.json()["message"]["content"].strip()
