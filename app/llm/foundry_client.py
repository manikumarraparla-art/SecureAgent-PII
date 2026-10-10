"""Foundry Local REST client below the existing secure service boundary.

Only local inference, no API key, no lifecycle/download API, no fallback.
Native tool capability is unavailable for the evaluated model/server; this
client cannot request or dispatch tools.
"""
from __future__ import annotations

import re

import httpx

from app.core.config import _parse_local_url, _validate_timeout
from app.core.exceptions import (ConfigurationError,
    LLMModelNotFoundError, LLMResponseError, LLMTimeoutError, LLMUnavailableError)
from app.llm.base import LLMHealth

# qwen3 (Foundry Local) may prefix its answer with a <think>...</think> block.
# The reasoning is never part of the answer, so it is removed here, at the
# provider boundary, before any application code, API, UI or log can see it.
_THINK_OPEN = re.compile(r'<think\b[^>]*>', re.IGNORECASE)
_THINK_CLOSE = re.compile(r'</think\s*>', re.IGNORECASE)


def final_answer(content: str) -> str:
    """Return only the final assistant answer from a Foundry completion.

    Text up to the last closing tag is reasoning (this also covers templates
    that omit the opening tag). An unclosed block cannot be separated from the
    answer reliably, so it fails closed instead of risking exposure.
    """
    closes = list(_THINK_CLOSE.finditer(content))
    answer = content[closes[-1].end():] if closes else content
    if _THINK_OPEN.search(answer):
        raise LLMResponseError('Foundry Local returned an incomplete reasoning block')
    answer = answer.strip()
    if not answer:
        raise LLMResponseError('Foundry Local returned no final answer')
    return answer


class FoundryLocalClient:
    provider_name = 'foundry'

    def __init__(self, base_url: str, model_name: str = 'qwen3-1.7b-generic-gpu',
                 connect_timeout: float = 5, request_timeout: float = 120,
                 transport: httpx.BaseTransport | None = None):
        base_url = _parse_local_url(base_url)
        if not isinstance(model_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', model_name):
            raise ConfigurationError('Invalid Foundry model identifier')
        _validate_timeout('Foundry connect timeout', connect_timeout)
        _validate_timeout('Foundry request timeout', request_timeout)
        self.model_name = model_name
        self._connect_timeout = connect_timeout
        self._client = httpx.Client(base_url=base_url,
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            transport=transport, trust_env=False, follow_redirects=False)

    @property
    def tool_calling_supported(self) -> bool:
        # CLI 0.10.3 + evaluated qwen3 model emits textual tool tags, not native
        # structured calls. Catalog advertising alone is insufficient evidence.
        return False

    def close(self):
        self._client.close()

    def _loaded_models(self) -> set[str]:
        try:
            response = self._client.get('/v1/models', timeout=self._connect_timeout)
            if response.status_code != 200 or len(response.content) > 1_000_000:
                raise ValueError
            models = response.json()['data']
            if not isinstance(models, list) or any(not isinstance(m.get('id'), str) for m in models):
                raise ValueError
            return {m['id'] for m in models}
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            raise LLMUnavailableError('Foundry Local model status unavailable') from None

    def check_health(self) -> LLMHealth:
        try:
            models = self._loaded_models()
        except LLMUnavailableError:
            return LLMHealth.UNAVAILABLE
        return LLMHealth.AVAILABLE if self.model_name in models else LLMHealth.MODEL_MISSING

    def _request(self, payload: dict) -> dict:
        # Require already loaded exact model. A cached model alone is insufficient.
        if self.model_name not in self._loaded_models():
            raise LLMModelNotFoundError('Configured Foundry model is not loaded')
        try:
            response = self._client.post('/v1/chat/completions', json=payload)
        except httpx.ConnectTimeout:
            raise LLMUnavailableError('Foundry Local unavailable') from None
        except httpx.TimeoutException:
            raise LLMTimeoutError('Foundry Local request timed out') from None
        except httpx.HTTPError:
            raise LLMUnavailableError('Foundry Local connection failed') from None
        if response.status_code == 404:
            raise LLMModelNotFoundError('Configured Foundry model unavailable')
        if response.status_code != 200:
            raise LLMResponseError('Foundry Local returned an invalid response')
        try:
            if len(response.content) > 1_000_000:
                raise ValueError
            body = response.json()
            choices = body['choices']
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            # Do not accept partial/truncated/filtered completions as successful.
            if choices[0]['finish_reason'] != 'stop':
                raise ValueError
            message = choices[0]['message']
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                raise ValueError
            return message
        except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
            raise LLMResponseError('Foundry Local returned a malformed completion') from None

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMResponseError('Prompt must be a non-empty string')
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        message = self._request({'model': self.model_name, 'messages': messages,
            'stream': False, 'temperature': 0, 'max_tokens': 2048})
        content = message.get('content')
        if not isinstance(content, str) or not content.strip() or len(content) > 32000 or message.get('tool_calls'):
            raise LLMResponseError('Foundry Local returned a non-text completion')
        return final_answer(content)

