import logging
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from omega.environment.conf_loader import omega_settings
import httpx

logger = logging.getLogger("GroqClient")

class GroqClient:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.api_key = omega_settings.groq_api_key
        if self.api_key:
            self.client = AsyncGroq(
                api_key=self.api_key,
                timeout=30,
                max_retries=0
            )
        else:
            self.client = None
            logger.warning("GROQ_API_KEY is missing! llm generation will fail")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=lambda retry_state: logger.warning(f"Groq API error, retrying.. (Attempt {retry_state.attempt_number})")
    )
    async def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            raise ValueError("Missing groq api key, Set GROQ_API_KEY in .env")

        response = await self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model_name,
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content