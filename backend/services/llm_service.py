"""Talks to the LLM. Two providers, chosen with LLM_PROVIDER:

  ollama  - a local model through Ollama's HTTP API (default)
  openai  - any OpenAI-compatible API (Groq's free tier by default)
"""
import requests

from backend import config


class LLMError(RuntimeError):
    """Raised with a human-readable message when the LLM can't be used."""


def generate(system_prompt, user_prompt):
    if config.LLM_PROVIDER == "ollama":
        return _generate_ollama(system_prompt, user_prompt)
    if config.LLM_PROVIDER == "openai":
        return _generate_openai(system_prompt, user_prompt)
    raise LLMError(
        f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'ollama' or 'openai'."
    )


def describe():
    """Short label for the UI, e.g. 'ollama / llama3.2'."""
    if config.LLM_PROVIDER == "ollama":
        return f"ollama / {config.OLLAMA_MODEL}"
    return f"{config.LLM_PROVIDER} / {config.LLM_MODEL}"


def _messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _generate_ollama(system_prompt, user_prompt):
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "messages": _messages(system_prompt, user_prompt),
        # Low temperature = factual, repeatable answers. num_ctx is how many
        # tokens the model can read at once.
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.LLM_TIMEOUT_SECONDS
        )
    except requests.ConnectionError as exc:
        raise LLMError(
            f"Cannot reach Ollama at {config.OLLAMA_URL}. Install it from ollama.com, "
            "start it (open the app or run `ollama serve`), then try again."
        ) from exc
    except requests.Timeout as exc:
        raise LLMError(
            f"The model did not answer within {config.LLM_TIMEOUT_SECONDS}s. "
            "Try again, or use a smaller model."
        ) from exc

    if response.status_code == 404:
        raise LLMError(
            f"Model '{config.OLLAMA_MODEL}' is not installed. "
            f"Run: ollama pull {config.OLLAMA_MODEL}"
        )
    if not response.ok:
        raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
    return response.json()["message"]["content"].strip()


def _generate_openai(system_prompt, user_prompt):
    if not config.LLM_API_KEY:
        raise LLMError(
            "No API key configured. Set LLM_API_KEY (for Groq: create a free key at "
            "console.groq.com) or switch back to LLM_PROVIDER=ollama."
        )
    payload = {
        "model": config.LLM_MODEL,
        "messages": _messages(system_prompt, user_prompt),
        "temperature": 0.1,
        "max_tokens": 900,
    }
    try:
        response = requests.post(
            f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError as exc:
        raise LLMError(f"Cannot reach the LLM API at {config.LLM_BASE_URL}.") from exc
    except requests.Timeout as exc:
        raise LLMError("The LLM API did not answer in time. Please try again.") from exc

    if response.status_code in (401, 403):
        raise LLMError("The LLM API rejected the API key. Check LLM_API_KEY.")
    if response.status_code == 404:
        raise LLMError(
            f"Model '{config.LLM_MODEL}' was not found at {config.LLM_BASE_URL}. "
            "Check LLM_MODEL against the provider's model list."
        )
    if response.status_code == 429:
        raise LLMError(
            "The free LLM quota is used up for the moment. Please try again in a minute."
        )
    if not response.ok:
        raise LLMError(f"LLM API error {response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]["content"].strip()
