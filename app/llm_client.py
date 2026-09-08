import httpx

from .config import settings


class LLMError(Exception):
    """Raised when the upstream LLM backend fails or misbehaves."""


class LLMClient:
    """Minimal client for any OpenAI-compatible chat completions API.

    Works with Ollama, OpenAI, Azure OpenAI (via compatible endpoint),
    LM Studio, vLLM, etc. — they all expose POST /chat/completions.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.timeout = timeout or settings.llm_timeout_seconds

    async def chat(self, prompt: str, model: str | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model or settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM backend timed out after {self.timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"LLM backend returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach LLM backend: {exc}") from exc

        data = response.json()
        try:
            return {
                "reply": data["choices"][0]["message"]["content"],
                "model": data.get("model", model or settings.llm_model),
            }
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected response format from LLM backend") from exc


class MockLLMClient(LLMClient):
    """Fake client used in mock mode and in unit tests."""

    async def chat(self, prompt: str, model: str | None = None) -> dict:
        return {
            "reply": f"[mock] You said: {prompt}",
            "model": "mock-model",
        }
