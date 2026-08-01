from __future__ import annotations
import os
from pathlib import Path
import httpx

ENV_PATH = Path(".env")
PROVIDERS = {
    "openai_compatible": ("OpenAI", "https://api.openai.com", "gpt-5.5"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash"),
    "anthropic": ("Anthropic", "https://api.anthropic.com", "claude-5-sonnet"),
}

def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values

def write_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    existing = read_env(path)
    existing.update(updates)
    lines = [f"{key}={value}" for key, value in sorted(existing.items())]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)

async def validate_llm_settings(provider: str, base_url: str, api_key: str, model: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if not api_key.strip():
        raise ValueError("An API key is required")
    if provider == "anthropic":
        url = base_url.rstrip("/") + "/v1/messages"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        payload = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "OK"}]}
    else:
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "OK"}]}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

    

def prompt_llm_settings(existing: dict[str, str]) -> dict[str, str]:
    print("Choose an LLM provider:")
    names = list(PROVIDERS)
    for index, provider in enumerate(names, 1):
        print(f"  {index}. {PROVIDERS[provider][0]}")
    selected = input(f"Provider [{existing.get('LLM_PROVIDER', 'openrouter')}]: ").strip()
    provider = names[int(selected) - 1] if selected.isdigit() and 1 <= int(selected) <= len(names) else selected or existing.get("LLM_PROVIDER", "openrouter")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    _, default_url, default_model = PROVIDERS[provider]
    base_url = input(f"Base URL [{existing.get('LLM_BASE_URL', default_url)}]: ").strip() or existing.get("LLM_BASE_URL", default_url)
    model = input(f"Model [{existing.get('LLM_MODEL', default_model)}]: ").strip() or existing.get("LLM_MODEL", default_model)
    api_key = input("API key: ").strip() or existing.get("LLM_API_KEY", "")
    return {"LLM_PROVIDER": provider, "LLM_BASE_URL": base_url, "LLM_MODEL": model, "LLM_API_KEY": api_key}