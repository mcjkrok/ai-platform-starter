from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables.

    Every field can be overridden with an env var of the same name,
    e.g. LLM_BASE_URL=http://ollama:11434/v1
    """

    app_name: str = "ai-platform-starter"
    app_version: str = "0.1.0"

    # Any OpenAI-compatible endpoint works here:
    # - Ollama:  http://localhost:11434/v1  (no key needed)
    # - OpenAI:  https://api.openai.com/v1  (requires LLM_API_KEY)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:1b"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 60.0

    # When true, the API returns canned responses without calling any LLM.
    # Useful for CI, demos and running on clusters without a model server.
    llm_mock: bool = False


settings = Settings()
