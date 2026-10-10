"""Provider selection and local endpoint discovery at application startup only."""
import json
import subprocess

from app.core.config import Settings, _parse_local_url
from app.core.exceptions import ConfigurationError
from app.llm.base import LLMClient
from app.llm.foundry_client import FoundryLocalClient
from app.llm.ollama_client import OllamaClient


def discover_foundry_endpoint() -> str:
    """Read the existing daemon's dynamic endpoint; never start/download models.

    Fixed CLI arguments, no shell and no request data. The response is untrusted
    configuration and must pass the same loopback validation as explicit URLs.
    """
    try:
        result = subprocess.run(
            ['foundry', 'server', 'status', '--output', 'json'],
            capture_output=True, text=True, timeout=10, check=True,
        )
        if len(result.stdout) > 16384:
            raise ValueError
        status = json.loads(result.stdout)
        urls = status['webUrls']
        if status['running'] is not True or not isinstance(urls, list) or len(urls) != 1:
            raise ValueError
        return _parse_local_url(urls[0])
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, ConfigurationError):
        raise ConfigurationError('Foundry local endpoint unavailable; start the local server and configure a loopback endpoint') from None


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == 'ollama':
        return OllamaClient.from_settings(settings)
    if settings.llm_provider == 'foundry':
        return FoundryLocalClient(
            base_url=settings.foundry_base_url or discover_foundry_endpoint(),
            model_name=settings.foundry_model,
            connect_timeout=settings.foundry_connect_timeout,
            request_timeout=settings.foundry_request_timeout,
        )
    raise ConfigurationError('Unsupported local LLM provider')
