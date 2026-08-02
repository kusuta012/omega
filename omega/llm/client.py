import logging
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, retry_if_exception_type
from omega.environment.conf_loader import omega_settings
from omega.llm.streaming import (
    StreamEvent,
    StreamProtocolError,
    ToolCall,
    ToolCallAccumulator,
    iter_sse_data,
    usage_from_mapping
)

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
    
    @abstractmethod
    def chat_with_tools_stream(
        self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

def is_retryable_http_error(exception: BaseException) -> bool:
    """Don't retry HTTP 400 or 404 errors - they are client errors like missing tool support."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code not in (400, 404)
    return isinstance(exception, httpx.RequestError)

def _provider_safe_tool_history(messages: list[dict]) -> list[dict]:
    safe_messages: list[dict] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            if message.get("role") != "tool":
                safe_messages.append(message)
            else:
                logger.warning("dropping orphaned tool result from provider history")
            index += 1
            continue

        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, list) or not tool_calls:
            logger.warning("dropping malformed assistant tool declaration")
            index += 1
            continue

        call_ids: set[str] = set()
        valid_declaration = True
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            call_id = str(call.get("id", "")) if isinstance(call, dict) else ""
            if not call_id or call_id in call_ids or not isinstance(function, dict):
                valid_declaration = False
                break
            if not isinstance(function.get("name"), str) or not function["name"]:
                valid_declaration = False
                break
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                valid_declaration = False
                break
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                valid_declaration = False
                break
            call_ids.add(call_id)

        result_messages = messages[index + 1:index + 1 + len(call_ids)]
        result_ids = [str(result.get("tool_call_id", "")) for result in result_messages]
        if (
            not valid_declaration
            or len(result_messages) != len(call_ids)
            or any(result.get("role") != "tool" for result in result_messages)
            or set(result_ids) != call_ids
            or len(set(result_ids)) != len(result_ids)
        ):
            logger.warning("dropping incomplete or out-of-order tool interaction from provider history")
            index += 1
            continue

        safe_messages.append(message)
        safe_messages.extend(result_messages)
        index += 1 + len(result_messages)
    return safe_messages


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self):
        self.base_url = omega_settings.llm_base_url.rstrip("/")
        self.api_key = omega_settings.llm_api_key
        self.model = omega_settings.llm_model
        if not self.api_key:
            logger.warning("LLM_API_KEY is missing! llm generation will fail")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.base_url.lower():
            if omega_settings.openrouter_app_url:
                headers["HTTP-Referer"] = omega_settings.openrouter_app_url
            if omega_settings.openrouter_app_title:
                headers["X-Title"] = omega_settings.openrouter_app_title
        return headers

    def _tool_payload(
        self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None
    ) -> dict[str, Any]:
        all_messages = _provider_safe_tool_history(messages)
        if tool_results:
            for result in tool_results:
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        if tools:
            payload["tools"] = tools
        return payload

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception(is_retryable_http_error),
        before_sleep=lambda retry_state: logger.warning(f"LLM API error, retrying.. (Attempt {retry_state.attempt_number})")
    )

    async def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
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
        payload: dict[str, Any] = {
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
            if e.response.status_code != 400:
                raise
            logger.warning("JSON mode not supported by this model/provider, falling back to regex extraction")
        except json.JSONDecodeError:
            logger.warning("Provider returned invalid JSON despite json mode, failling back to refex extraction")

        fallback = system_prompt + "\n\n CRITICAL: You MUST respond with only valid JSON, No markdown, no explanations"
        payload["messages"][0]["content"] = fallback
        payload.pop("response_format", None)

        data = await self._call_api(payload)
        content = data["choices"][0]["message"]["content"]

        match = re.search(r"(\{.*?\}|\[.*?\])", content, re.DOTALL)
        if match:
            extracted = match.group(1)
            try:
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to generate parseable JSON, raw output {content}")

    async def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None) -> ChatResponse:
        try:
            data = await self._call_api(self._tool_payload(messages, tools, tool_results))
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404) and tools:
                logger.error(f"Tool calling failed ({e.response.status_code}). Likely unspported model: {e.response.text}")
                return ChatResponse(
                    content="Error: The currently selected LLM model does not support native tool-calling. Please switch to a supported model.",
                )
            raise
            
        msg = data["choices"][0]["message"]

        tool_calls = self._parse_tool_calls(msg.get("tool_calls") or [])
        return ChatResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=data["choices"][0].get("finish_reason", "stop")
        )
    
    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for raw_call in raw_calls:
            function = raw_call.get("function") or {}
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(
                    f"Provider returned invalid JSON for tool {function.get('name', '<unknown>')}: {e.msg} "
                ) from e
            if not isinstance(arguments, dict):
                raise StreamProtocolError("Provider returned non-object tool arguments")
            parsed.append(ToolCall(str(raw_call["id"]), str(function["name"]), arguments))
        return parsed

    async def chat_with_tools_stream(self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None) -> AsyncIterator[StreamEvent]:
        payload = self._tool_payload(messages, tools, tool_results)
        payload["stream"] = True
        accumulator = ToolCallAccumulator()
        finish_reason = "stop"
        terminal_event_received = False
        usage: dict[str, int] = {}
        emitted_content = False

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    response.raise_for_status()
                    async for data_line in iter_sse_data(response):
                        if data_line == "[DONE]":
                            break
                        try:
                            event = json.loads(data_line)
                        except json.JSONDecodeError:
                            logger.warning("ignoring malformed Openai-compatible sse payload")
                            continue

                        usage.update(usage_from_mapping(event.get("usage")))
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            emitted_content = True
                            yield StreamEvent.text_delta(content)
                        for raw_call in delta.get("tool_calls") or []:
                            accumulator.add_openai_delta(raw_call)
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                            terminal_event_received = True
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 404) and tools:
                    logger.error(f"streaming tool calling failed {e.response.status_code}:{e.response.text}")
                    raise StreamProtocolError(
                        "The currently selected LLM model does not support native tool-calling"
                        "Please switch to a supported model"
                    ) from e 
                raise
        
        if not terminal_event_received:
            raise StreamProtocolError(
                "OpenAI compatible stream ended before a provider completion event"
            )

        completed_tool_calls = accumulator.finalize()
        for tool_call in completed_tool_calls:
            yield StreamEvent.completed_tool_call(tool_call)
        if not emitted_content and not completed_tool_calls:
            logger.warning("openai compatible stream completed without text or tool calls")
        yield StreamEvent.message_end(finish_reason, usage)
        

class AnthropicProvider(LLMProvider):
    API_URL = "https://api.anthropic.com/v1/messages"
    def __init__(self):
        self.api_key = omega_settings.llm_api_key
        self.model = omega_settings.llm_model
        if not self.api_key:
            logger.warning("LLM_API_KEY is missing! llm generation will fail")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=lambda rs: logger.warning(f"LLM API error, retrying.. (Attempt {rs.attempt_number})")
    )
    async def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.API_URL, json=payload, headers=self._headers())
            response.raise_for_status()
            return response.json()

    def _tool_payload(
        self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None
    ) -> dict[str, Any]:
        anthropic_tools  = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"].get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for tool in tools
        ]

        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                anthropic_messages.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for message in _provider_safe_tool_history(messages):
            role = message["role"]
            if role == "system":
                system_parts.append(message["content"])
                continue
            if role == "tool":
                pending_tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                })
                continue
            flush_tool_results()
            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for raw_call in message["tool_calls"]:
                    function = raw_call.get("function") or {}
                    raw_arguments = function.get("arguments", "{}")
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError as e:
                        raise StreamProtocolError("stored assistant tool call has invalid JSON") from e
                    if not isinstance(arguments, dict):
                        raise StreamProtocolError("stored assistant call has non-object arguments")
                    blocks.append({
                        "type": "tool_use",
                        "id": raw_call["id"],
                        "name": function["name"],
                        "input": arguments,
                    })
                anthropic_messages.append({"role": "assistant", "content": blocks})
                continue
            
            anthropic_messages.append({"role": role, "content": message["content"]})

        flush_tool_results()

        if tool_results:
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": result["tool_call_id"],
                    "content": result["content"],
                } for result in tool_results],
            })
        
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
        return payload
        
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
        data = await self._call_api(self._tool_payload(messages, tools, tool_results))
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data["content"]:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                input_value = block.get("input", {})
                if not isinstance(input_value, dict):
                    raise StreamProtocolError("anthropic returned non-object tool arguments")
                tool_calls.append(ToolCall(block["id"], block["name"], input_value))
        return ChatResponse(content="".join(text_parts) or None, tool_calls=tool_calls, finish_reason=data.get("stop_reason", "end_turn"))

    async def chat_with_tools_stream(self, messages: list[dict], tools: list[dict], tool_results: list[dict] | None = None) -> AsyncIterator[StreamEvent]:
        payload = self._tool_payload(messages, tools, tool_results)
        payload["stream"] = True
        accumulator = ToolCallAccumulator()
        finish_reason = "end_turn"
        terminal_event_received = False
        usage: dict[str, int] = {}

        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15)) as client:
            async with client.stream("POST", self.API_URL, json=payload, headers=self._headers()) as response:
                response.raise_for_status()
                async for data_line in iter_sse_data(response):
                    try:
                        event = json.loads(data_line)
                    except json.JSONDecodeError:
                        logger.warning("Ignoring malformed anthropic sse payload")
                        continue
                    
                    event_type = event.get("type")
                    if event_type == "message_start":
                        usage.update(usage_from_mapping((event.get("message") or {}).get("usage")))
                    elif event_type == "content_block_start":
                        block = event.get("content_block") or {}
                        index = event.get("index")
                        if block.get("type") == "tool_use":
                            if not isinstance(index, int):
                                raise StreamProtocolError("anthropic tool block did not include an integer index")
                            accumulator.start_anthropic_tool(index, block)
                    elif event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        index = event.get("index")
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield StreamEvent.text_delta(str(delta["text"]))
                        elif delta.get("type") == "input_json_delta":
                            if not isinstance(index, int):
                                raise StreamProtocolError("Anthropic tool JSON delta did not include an integer index")
                            partial_json = delta.get("partial_json", "")
                            if not isinstance(partial_json, str):
                                raise StreamProtocolError("anthropic tool JSON delta was not text")
                            accumulator.add_anthropic_json_delta(index, partial_json)
                    elif event_type == "message_stop":
                        terminal_event_received = True
                    elif event_type == "message_delta":
                        delta = event.get("delta") or {}
                        if delta.get("stop_reason"):
                            finish_reason = str(delta["stop_reason"])
                        usage.update(usage_from_mapping(event.get("usage")))

        if not terminal_event_received:
            raise StreamProtocolError("Anthropic stream ended before a message_stop event")
        for tool_call in accumulator.finalize():
            yield StreamEvent.completed_tool_call(tool_call)
        yield StreamEvent.message_end(finish_reason, usage)


def get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        provider_type = omega_settings.llm_provider.lower()
        if provider_type in {"openai_compatible", "openrouter"}:
            _llm_provider = OpenAICompatibleProvider()
        elif provider_type == "anthropic":
            _llm_provider = AnthropicProvider()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}, use 'openai_compatible' or 'anthropic'..")
    return _llm_provider

_llm_provider: LLMProvider | None = None