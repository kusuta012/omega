from __future__ import annotations
import asyncio
from omega.cli.configuration import prompt_llm_settings, read_env, validate_llm_settings, write_env
from omega.llm.provider_specs import sanitize_provider_error

def run_config() -> int:
    existing = read_env()
    try:
        settings = prompt_llm_settings(existing)
        print("Validating LLM settings..")
        asyncio.run(validate_llm_settings(settings["LLM_PROVIDER"], settings["LLM_BASE_URL"], settings["LLM_API_KEY"], settings["LLM_MODEL"]))
        write_env(settings)
    except Exception as ex:
        print(f"Configuration was not saved: {sanitize_provider_error(ex)}")
        return 1
    print("Configuration saved. It will apply on the next omega launch")
    return 0