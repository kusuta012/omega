from dataclasses import dataclass
from multiprocessing import Value
from unicodedata import normalize
from urllib.parse import urlparse

import httpx

@dataclass(frozen=True)
class ProviderSpec:
    name: str
    display_name: str
    base_url: str
    model: str
    protocol: str

@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    spec: ProviderSpec

PROVIDER_SPECS = {
    "openai_compatible": ProviderSpec(
        "openai_compatible", "OpenAI-compatible", "https://api.openai.com/v1", "gpt-5.6-terra", "openai"
    ),
    "openrouter": ProviderSpec(
        "openrouter", "Openrouter", "https://openrouter.ai/api/v1", "nvidia/nemotron-3-ultra-550b-a55b:free", "openai",
    ),
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic", "https://api.anthropic.com", "claude-5-sonnet", "anthropic"
    ),
}
DEFAULT_PROVIDER = "openrouter"

def normalize_provider_settings(provider: str, base_url: str, api_key: str, model: str) -> ProviderSettings:
    provider_name = provider.strip().lower()
    spec = PROVIDER_SPECS.get(provider_name)
    if spec is None:
        raise ValueError(f"Unsupported provider: {provider}")
    normalized_url = base_url.strip().rstrip("/")
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("LLM base URL must a credential-free http or https URL")
    if not api_key.strip():
        raise ValueError("An API key is required")
    if not model.strip():
        raise ValueError("An LLM model is required")
    return ProviderSettings(provider_name, normalized_url, api_key.strip(), model.strip(), spec)

def sanitize_provider_error(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"Provider returned HTTP {error.response.status_code}"
    if isinstance(error, httpx.RequestError):
        return "Provider request failed"
    if isinstance(error, ValueError):
        return str(error)
    return "Provider validation failed"