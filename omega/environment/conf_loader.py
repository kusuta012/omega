from pydantic_settings import BaseSettings
from omega.llm.provider_specs import sanitize_provider_error, DEFAULT_PROVIDER, PROVIDER_SPECS
_DEFAULT_PROVIDER = PROVIDER_SPECS[DEFAULT_PROVIDER]

class OmegaSettings(BaseSettings):
    database_url: str = "postgresql://omega:pieistlecker@localhost:5432/omega_db"
    llm_provider: str = DEFAULT_PROVIDER
    llm_base_url: str = _DEFAULT_PROVIDER.base_url
    llm_api_key: str = ""
    llm_model: str = _DEFAULT_PROVIDER.model
    openrouter_app_url: str = ""
    openrouter_app_title: str = ""
    session_token_budget: int = 64000
    session_compression_ratio: float = 0.50
    session_emergency_ratio: float = 0.85
    memory_inject_recent_summaries: int = 2
    memory_dir: str = "./omega_memory"
    tail_preserve_tokens: int = 20000
    embedding_model: str = "all-MiniLM-L6-v2"
    profile_inference_enabled: bool = True
    profile_inference_min_messages: int = 20
    max_tool_rounds_per_turn: int = 5
    max_tool_calls_per_turn: int = 8
    tool_result_chunk_chars: int = 2000
    max_turn_overflow_chars: int = 16000
    max_turn_execution_items: int = 6
    max_turn_execution_text: int = 320
    class Config:
        env_file = ".env"
        extra = "ignore"

omega_settings = OmegaSettings()