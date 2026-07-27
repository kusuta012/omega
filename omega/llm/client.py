import logging
import json
import re
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from omega.environment.conf_loader import omega_settings
import httpx
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ChatResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

logger = logging.getLogger("LLMProvider")

class LLMProvider(ABC):
    @abstractmethod
    async def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        pass

    @abstractmethod
    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None
    ) -> ChatResponse:
        pass

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self):
        self.base_url = omega_settings.llm_base_url.rstrip("/")
        self.api_key = omega_settings.llm_api_key
        self.model = omega_settings.llm_model
        if not self.api_key:
            logger.warning("LLM_API_KEY is missing! llm generation will fail")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=lambda retry_state: logger.warning(f"LLM API error, retrying.. (Attempt {retry_state.attempt_number})")
    )
    async def _call_api(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        data = await self._call_api(payload)
        return data["choices"][0]["message"]["content"]

    async def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "json_object"}
        }

        try:
            data = await self._call_api(payload)
            content = data["choices"][0]["message"]["content"]
            json.loads(content)
            return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                logger.warning("JSON mode not supported by this model/provider, falling back to regex extraction")
            else:
                raise e
        except json.JSONDecodeError:
            logger.warning("Provider returned invalid JSON despite json mode, failling back to refex extraction")

        fallback = system_prompt + "\n\n CRITICAL: You MUST respond with only valid JSON, No markdown, no explanations"
        payload["messages"][0]["content"] = fallback
        payload.pop("response_format", None)

        data = await self._call_api(payload)
        content = data["choices"][0]["message"]["content"]

        match = re.search(r'(\{.*?\}|\[.*?\])', content, re.DOTALL)
        if match:
            extracted = match.group(1)
            try:
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to generate parseable JSON, raw output {content}")

    async def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None) -> ChatResponse:
        all_messages = list(messages)
        if tool_results:
            for tr in tool_results:
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"]
                })
            
        payload = {
            "model": self.model,
            "messages": all_messages,
            "temperature": 0.3,
            "max_tokens": 2048
        }
        if tools:
            payload["tools"] = tools

        data = await self._call_api(payload)
        msg = data["choices"][0]["message"]

        tool_calls = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"])
                ))

        return ChatResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=data["choices"][0].get("finish_reason", "stop")
        )

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = omega_settings.llm_api_key
        self.model = omega_settings.llm_model
        if not self.api_key:
            logger.warning("LLM_API_KEY is missing! llm generation will fail")

        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=lambda rs: logger.warning(f"LLM API error, retrying.. (Attempt {rs.attempt_number})")
    )
    async def _call_api(self, payload: dict) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        data = await self._call_api(payload)
        return next(
            block["text"] for block in data["content"]
            if block["type"] == "text"
        )

    async def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
            "output_config": {
                "format": {
                    "type": "json_object"
                }
            }
        }
        data = await self._call_api(payload)
        content = next(
            block["text"] for block in data["content"]
            if block["type"] == "text"
        )
        json.loads(content)
        return content

    async def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None) -> ChatResponse:
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})
            })

        system_text = None
        anthropic_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "tool":
                anthropic_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", "unknown"),
                        "content": m["content"]
                    }]
                })
            else:
                anthropic_msgs.append({"role": m["role"], "content": m["content"]})

        if tool_results:
            tool_result_blocks = []
            for tr in tool_results:
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tr["tool_call_id"],
                    "content": tr["content"]
                })
            anthropic_msgs.append({"role": "user", "content": tool_result_blocks})

        payload = {
            "model": self.model,
            "messages": anthropic_msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }
        if system_text:
            payload["system"] = system_text
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        data = await self._call_api(payload)
        content_text = None
        tool_calls = []
        for block in data["content"]:
            if block["type"] == "text":
                content_text = block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block["input"]
                ))

        return ChatResponse(content=content_text, tool_calls=tool_calls, finish_reason=data.get("stop_reason", "end_turn"))

def get_llm_provider() -> LLMProvider:
    provider_type = omega_settings.llm_provider.lower()
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider()
    elif provider_type == "anthropic":
        return AnthropicProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}, use 'openai_compatible' or 'antrhopic'..")