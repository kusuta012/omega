from pydantic_settings import BaseSettings

class OmegaSettings(BaseSettings):
    database_url: str = "postgresql://omega:pieistlecker@localhost:5432/omega_db"
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"

    class Config:
        env_file = ".env"
        extra = "ignore"

omega_settings = OmegaSettings()