"""Client for a local Ollama server, implementing `LLMClient`.

Security properties:
- Only accepts already-sanitized strings (it is called exclusively by
  `SecureLLMService`, after the PII Security Gateway).
- Never logs prompt or response contents.
- Never pulls/downloads models and never falls back to another model.
- Ignores proxy environment variables (`trust_env=False`), so prompts cannot
  be routed off the machine by an inherited HTTP(S)_PROXY setting.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.config import Settings, _parse_local_url, _validate_timeout
from app.core.exceptions import (
    LLMModelNotFoundError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.base import LLMHealth, ToolCall, ModelTurn

logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous client for Ollama's local `/api/chat` endpoint."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b",
        connect_timeout: float = 5.0,
        request_timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Args:
            base_url: Ollama server URL (validated here and by `Settings`).
            model_name: Exact model tag to use. No fallback is attempted.
            connect_timeout: Seconds to establish a connection; also bounds
                the reachability check.
            request_timeout: Seconds to wait for a generation to complete.
            transport: Optional httpx transport (used by tests to intercept
                the Ollama boundary).
        """
        base_url = _parse_local_url(base_url)
        _validate_timeout('Ollama connect timeout', connect_timeout)
        _validate_timeout('Ollama request timeout', request_timeout)
        self.model_name = model_name
        self._connect_timeout = connect_timeout
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        return cls(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model,
            connect_timeout=settings.ollama_connect_timeout,
            request_timeout=settings.ollama_request_timeout,
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ health

    def _installed_models(self) -> set[str]:
        """Return locally installed model tags; raise if Ollama is unreachable."""
        try:
            response = self._client.get("/api/tags", timeout=self._connect_timeout)
        except httpx.HTTPError:
            raise LLMUnavailableError("Ollama server is not reachable") from None
        if response.status_code != 200:
            raise LLMUnavailableError(f"Ollama health check returned HTTP {response.status_code}")
        try:
            return {m["name"] for m in response.json().get("models", [])}
        except (ValueError, KeyError, TypeError, AttributeError):
            raise LLMUnavailableError("Ollama returned an unreadable model list") from None

    def _model_installed(self, installed: set[str]) -> bool:
        if self.model_name in installed:
            return True
        # Ollama reports untagged models as "<name>:latest".
        return ":" not in self.model_name and f"{self.model_name}:latest" in installed

    def check_health(self) -> LLMHealth:
        """Report whether Ollama is reachable and the configured model is installed."""
        try:
            installed = self._installed_models()
        except LLMUnavailableError:
            return LLMHealth.UNAVAILABLE
        return LLMHealth.AVAILABLE if self._model_installed(installed) else LLMHealth.MODEL_MISSING

    # -------------------------------------------------------------- generation

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Send a sanitized prompt to the configured model and return its reply.

        Raises:
            LLMUnavailableError: Ollama cannot be reached.
            LLMModelNotFoundError: the configured model is not installed.
            LLMTimeoutError: generation exceeded the request timeout.
            LLMResponseError: Ollama returned an error or malformed response.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMResponseError("Prompt must be a non-empty string")

        # Verify reachability and model presence first: no auto-pull, no fallback.
        if not self._model_installed(self._installed_models()):
            raise LLMModelNotFoundError(
                f"Model {self.model_name!r} is not installed in the local Ollama server"
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {"model": self.model_name, "messages": messages, "stream": False}

        started = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.ConnectTimeout:
            raise LLMUnavailableError("Ollama server is not reachable") from None
        except httpx.TimeoutException:
            raise LLMTimeoutError("Ollama did not respond within the request timeout") from None
        except httpx.HTTPError:
            raise LLMUnavailableError("Ollama connection failed") from None

        if response.status_code == 404:
            raise LLMModelNotFoundError(f"Model {self.model_name!r} is not available")
        if response.status_code != 200:
            # The body is not included: it is backend detail, not client-safe.
            raise LLMResponseError(f"Ollama returned HTTP {response.status_code}")

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMResponseError("Ollama returned a malformed response") from None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama returned a non-text response")

        logger.info(
            "Ollama generation completed model=%s latency_ms=%.0f",
            self.model_name, (time.perf_counter() - started) * 1000,
        )
        return content


    def complete(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        """Native structured tools, proposed only. Never dispatches a function."""
        if not messages or not isinstance(messages,list) or not isinstance(tools,list):
            raise LLMResponseError("Invalid structured request")
        if not self._model_installed(self._installed_models()):
            raise LLMModelNotFoundError("Configured model missing")
        try:
            response=self._client.post("/api/chat",json={"model":self.model_name,
                "messages":messages,"tools":tools,"stream":False,"options":{"temperature":0,"num_predict":1024}})
        except httpx.ConnectTimeout:
            raise LLMUnavailableError("Local model unavailable") from None
        except httpx.TimeoutException:
            raise LLMTimeoutError("Local model timeout") from None
        except httpx.HTTPError:
            raise LLMUnavailableError("Local model unavailable") from None
        if response.status_code==404: raise LLMModelNotFoundError("Configured model missing")
        if response.status_code!=200: raise LLMResponseError("Invalid local model response")
        try:
            if len(response.content)>1_000_000: raise ValueError("Oversized model response")
            message=response.json()['message']
            if message.get('role','assistant')!='assistant': raise ValueError("Invalid model role")
            content=message.get('content','')
            calls=message.get('tool_calls',[])
            if not isinstance(content,str) or len(content)>32000 or not isinstance(calls,list) or len(calls)>10:
                raise ValueError("Invalid model response")
            parsed=[]
            for call in calls:
                function=call['function']
                name,args=function['name'],function['arguments']
                if not isinstance(name,str) or len(name)>128 or type(args) is not dict:
                    raise ValueError("Invalid model call")
                parsed.append(ToolCall(name,args))
            return ModelTurn(content,tuple(parsed))
        except (ValueError,TypeError,KeyError,AttributeError,RecursionError):
            raise LLMResponseError("Malformed structured model response") from None
