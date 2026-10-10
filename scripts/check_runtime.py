"""Read-only release/provider check. Run: python -m scripts.check_runtime.

Never starts, stops or reconfigures a process, and never submits user data.
"""
import argparse
import json

import httpx
from pydantic import ValidationError

from app.version import VERSION
from secureagent_ui.api_client import UIFailure, local_backend_url
from secureagent_ui.contracts import HealthResponse


def inspect_runtime(client, provider):
    """Check the live contract, not just liveness. Errors contain no remote text."""
    try:
        health_response = client.get('/health')
        schema_response = client.get('/openapi.json')
        if len(health_response.content) > 4096 or len(schema_response.content) > 2_000_000:
            raise ValueError
        if health_response.status_code != 200 or schema_response.status_code != 200:
            raise ValueError
        health = HealthResponse.model_validate(health_response.json())
        schema = schema_response.json()
        if (health.version != VERSION or schema['info']['version'] != VERSION
                or health.provider != provider or health.status != 'healthy'
                or health.llm != 'available'):
            raise ValueError
        for name in ('ChatRequest', 'RAGChatRequest', 'AgentChatRequest'):
            model = schema['components']['schemas'][name]
            if model.get('additionalProperties') is not False:
                raise ValueError
            if model['properties']['include_orchestration']['type'] != 'boolean':
                raise ValueError
        return {'result': 'PASS', 'version': health.version, 'provider': health.provider,
                'model': health.model, 'tool_calling': health.tool_calling, 'request_contract': 'compatible'}
    except (httpx.HTTPError, ValidationError, ValueError, TypeError, KeyError, AttributeError):
        raise ValueError('Runtime check failed. Check the port owner, restart this checkout if stale, '
                         'and verify the selected provider/model is loaded.') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', default='http://127.0.0.1:8000')
    parser.add_argument('--provider', choices=('ollama', 'foundry'), default='ollama')
    args = parser.parse_args()
    try:
        url = local_backend_url(args.backend_url)
        with httpx.Client(base_url=url, trust_env=False, follow_redirects=False, timeout=5) as client:
            result = inspect_runtime(client, args.provider)
    except (UIFailure, ValueError) as exc:
        print(json.dumps({'result': 'FAIL', 'message': str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
