from pydantic_settings import BaseSettings

class OmegaSettings(BaseSettings):
    database_url: str = "postgresql://omega:pieistlecker@localhost:5432/omega_db"

    class Config:
        env_file = ".env"

omega_settings = OmegaSettings()