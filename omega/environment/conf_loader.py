from pydantic_settings import BaseSettings

class OmegaSettings(BaseSettings):
    database_url: str = "postgresql://omega:pieistlecker@localhost:5432/omega_db"
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    openrouter_app_url: str = ""
    openrouter_app_title: str = ""
    session_token_budget: int = 64000
    session_compression_ratio: float = 0.50
    session_emergency_ratio: float = 0.85
    memory_inject_recent_summaries: int = 2
    memory_dir: str = "./omega_memory"
    tail_preserve_tokens: int = 20000
    
    class Config:
        env_file = ".env"
        extra = "ignore"

omega_settings = OmegaSettings()