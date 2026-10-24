import unittest
from unittest.mock import MagicMock, patch

from backend import config
from backend.services import llm_service
from backend.services.llm_service import LLMError


def fake_response(status=200, json_body=None, text=""):
    response = MagicMock()
    response.status_code = status
    response.ok = status < 400
    response.text = text
    response.json.return_value = json_body or {}
    return response


class LLMServiceTests(unittest.TestCase):
    def test_openai_success_sends_key_and_returns_text(self):
        body = {"choices": [{"message": {"content": "  hello  "}}]}
        with patch.object(config, "LLM_PROVIDER", "openai"), \
             patch.object(config, "LLM_API_KEY", "k123"), \
             patch("requests.post", return_value=fake_response(json_body=body)) as post:
            self.assertEqual(llm_service.generate("sys", "user"), "hello")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer k123")
        self.assertTrue(post.call_args.args[0].endswith("/chat/completions"))

    def test_openai_without_key_is_a_clear_error(self):
        with patch.object(config, "LLM_PROVIDER", "openai"), patch.object(config, "LLM_API_KEY", ""):
            with self.assertRaises(LLMError) as ctx:
                llm_service.generate("s", "u")
        self.assertIn("API key", str(ctx.exception))

    def test_openai_error_codes_are_mapped(self):
        cases = {401: "rejected", 404: "not found", 429: "quota"}
        for status, word in cases.items():
            with patch.object(config, "LLM_PROVIDER", "openai"), \
                 patch.object(config, "LLM_API_KEY", "k"), \
                 patch("requests.post", return_value=fake_response(status=status)):
                with self.assertRaises(LLMError) as ctx:
                    llm_service.generate("s", "u")
            self.assertIn(word, str(ctx.exception))

    def test_ollama_not_running_message(self):
        import requests

        with patch.object(config, "LLM_PROVIDER", "ollama"), \
             patch("requests.post", side_effect=requests.ConnectionError()):
            with self.assertRaises(LLMError) as ctx:
                llm_service.generate("s", "u")
        self.assertIn("ollama serve", str(ctx.exception))

    def test_ollama_missing_model_message(self):
        with patch.object(config, "LLM_PROVIDER", "ollama"), \
             patch("requests.post", return_value=fake_response(status=404)):
            with self.assertRaises(LLMError) as ctx:
                llm_service.generate("s", "u")
        self.assertIn("ollama pull", str(ctx.exception))

    def test_unknown_provider(self):
        with patch.object(config, "LLM_PROVIDER", "nope"):
            with self.assertRaises(LLMError):
                llm_service.generate("s", "u")


if __name__ == "__main__":
    unittest.main()
