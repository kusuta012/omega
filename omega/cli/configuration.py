from __future__ import annotations
import os
from pathlib import Path
import httpx
from omega.llm.provider_specs import (DEFAULT_PROVIDER, PROVIDER_SPECS, normalize_provider_settings)

ENV_PATH = Path(".env")
PROVIDERS = {
    name: (spec.display_name, spec.base_url, spec.model)
    for name, spec in PROVIDER_SPECS.items()
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
    settings = normalize_provider_settings(provider, base_url, api_key, model)
    if settings.spec.protocol == "anthropic":
        url = settings.base_url + "/v1/messages"
        headers = {"x-api-key": settings.api_key, "anthropic-version": "2023-06-01"}
        payload = {"model": settings.model, "max_tokens": 1, "messages": [{"role": "user", "content": "OK"}]}
    else:
        url = settings.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {settings.api_key}"}
        payload = {"model": settings.model, "max_tokens": 1, "messages": [{"role": "user", "content": "OK"}]}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

    

def prompt_llm_settings(existing: dict[str, str]) -> dict[str, str]:
    print("Choose an LLM provider:")
    names = list(PROVIDERS)
    for index, provider in enumerate(names, 1):
        print(f"  {index}. {PROVIDERS[provider][0]}")
    selected = input(f"Provider [{existing.get('LLM_PROVIDER', DEFAULT_PROVIDER)}]: ").strip()
    provider = (names[int(selected) - 1] if selected.isdigit() and 1 <= int(selected) <= len(names) else selected or existing.get("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    _, default_url, default_model = PROVIDERS[provider]
    previous_provider = existing.get("LLM_PROVIDER", "").strip().lower()
    default_base_url = existing.get("LLM_BASE_URL", default_url) if previous_provider == provider else default_url
    default_model_value = existing.get("LLM_MODEL", default_model) if previous_provider == provider else default_model
    base_url = input(f"Base URL [{default_base_url}]: ").strip() or default_base_url
    model = input(f"Model [{default_model_value}]: ").strip() or default_model_value
    api_key = input("API key: ").strip() or existing.get("LLM_API_KEY", "")
    return {"LLM_PROVIDER": provider, "LLM_BASE_URL": base_url, "LLM_MODEL": model, "LLM_API_KEY": api_key}