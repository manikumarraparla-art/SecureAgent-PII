"""Architecture guards: exactly one controlled path to the LLM.

    Application -> SecureLLMService -> PIISecurityGateway -> LLMClient

These static checks fail if any code outside the secure service calls an
LLM, or if the API layer can reach an LLM client directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
SECURE_SERVICE = APP / "security" / "secure_llm.py"
COMPOSITION_ROOT = APP / "main.py"


def _modules() -> dict[Path, ast.Module]:
    return {p: ast.parse(p.read_text(), filename=str(p)) for p in APP.rglob("*.py")}


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_only_secure_service_calls_llm_generate():
    offenders = []
    for path, tree in _modules().items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "generate" and path != SECURE_SERVICE):
                offenders.append(f"{path.relative_to(APP)}:{node.lineno}")
    assert offenders == [], f"LLM invoked outside SecureLLMService: {offenders}"


def test_api_layer_cannot_reach_llm_clients_or_http():
    forbidden = ("app.llm.ollama_client", "httpx", "app.llm.base.LLMClient")
    for path, tree in _modules().items():
        if APP / "api" in path.parents:
            imported = _imported_names(tree)
            hits = [f for f in forbidden if any(i == f or i.startswith(f + ".") for i in imported)]
            assert hits == [], f"{path.name} imports {hits}"


def test_provider_clients_constructed_only_in_composition_layer():
    for path, tree in _modules().items():
        if path in (COMPOSITION_ROOT, APP / "llm" / "ollama_client.py", APP / "llm" / "foundry_client.py", APP / "llm" / "factory.py"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ("OllamaClient", "FoundryLocalClient", "create_llm_client"):
                raise AssertionError(f"Provider client referenced in {path.relative_to(APP)}")


def test_composition_root_hands_llm_only_to_secure_service():
    source = COMPOSITION_ROOT.read_text()
    assert "llm=create_llm_client(settings)" in source
    assert "app.state.llm" not in source


def test_no_model_download_or_cloud_sdks():
    banned_strings = ("/api/pull", "/api/create", "openai", "azure")
    banned_imports = ("langchain", "llama_index", "openai", "azure")
    for path, tree in _modules().items():
        source = path.read_text().lower()
        for s in banned_strings:
            assert s not in source, f"{path.name} contains {s!r}"
        for name in _imported_names(tree):
            assert not name.startswith(banned_imports), f"{path.name} imports {name}"

def test_rag_architecture_enforcement():
    root=APP / "rag"
    for path in root.glob('*.py'):
        tree=ast.parse(path.read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(i and i.startswith('app.llm') for i in imports)
        assert 'OllamaClient' not in path.read_text()
        if path.name not in ('embeddings.py',):
            assert not any(isinstance(n,ast.Import) and any(x.name=='httpx' for x in n.names) for n in ast.walk(tree))
    assert '/api/chat' not in (root / "embeddings.py").read_text()
    assert '/api/generate' not in (root / "embeddings.py").read_text()


def test_native_model_calls_stay_in_secure_service():
    for path,tree in _modules().items():
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='complete':
                assert path==SECURE_SERVICE, f'Native model invocation outside security: {path}'


def test_only_mcp_host_dispatches_tools():
    # Phase 8: registry handlers run only inside the local MCP servers' host,
    # which the agent reaches solely through ToolGateway -> MCP gateway.
    for path,tree in _modules().items():
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='dispatch':
                assert path==APP/'mcp'/'host.py'


def test_api_and_agent_cannot_import_concrete_tools():
    for path,tree in _modules().items():
        if APP/'api' in path.parents or (APP/'agent' in path.parents and path.name!='tool_registry.py'):
            imports=_imported_names(tree)
            forbidden=('app.tools','app.agent.tool_registry','httpx','subprocess','app.llm.ollama_client')
            assert not any(any(name==f or name.startswith(f+'.') for f in forbidden) for name in imports)


def test_tools_cannot_reach_models_or_external_io():
    for path,tree in _modules().items():
        if APP/'tools' in path.parents or path==APP/'agent'/'planner.py' or APP/'graph' in path.parents:
            imports=_imported_names(tree)
            forbidden=('app.llm.ollama_client','httpx','requests','subprocess','os','sqlite3','socket')
            assert not any(any(name==f or name.startswith(f+'.') for f in forbidden) for name in imports)
            assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','open') for n in ast.walk(tree))


def test_ui_is_only_a_public_api_client():
    forbidden=('app.security','app.llm','app.rag','app.agent','app.tools','app.audit','app.core','app.main','app.models')
    ui_package=APP.parent/'secureagent_ui'
    modules=_modules() | {p:ast.parse(p.read_text()) for p in ui_package.glob('*.py')}
    for path,tree in modules.items():
        if APP/'ui' in path.parents or ui_package in path.parents:
            imports=_imported_names(tree)
            assert not any(any(name==f or name.startswith(f+'.') for f in forbidden) for name in imports)
            assert not any(isinstance(n,ast.Name) and n.id in ('OllamaClient','SecureLLMService','PIIDetector','PIITokenizer','Retriever','ToolGateway') for n in ast.walk(tree))
            if path.name!='api_client.py':
                assert not any(any(name==f or name.startswith(f+'.') for f in ('httpx','requests','socket')) for name in imports)


def test_ui_has_no_persistence_or_analytics_calls():
    ui_package=APP.parent/'secureagent_ui'
    modules=_modules() | {p:ast.parse(p.read_text()) for p in ui_package.glob('*.py')}
    for path,tree in modules.items():
        if APP/'ui' in path.parents or ui_package in path.parents:
            imports=_imported_names(tree)
            assert not any(any(name==f or name.startswith(f+'.') for f in ('sqlite3','pickle','shelve','logging','subprocess')) for name in imports)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    assert node.func.attr not in ('cache_data','cache_resource','to_csv','to_json','write_text','write_bytes')


def test_ui_network_endpoints_are_fixed_fastapi_routes():
    tree=ast.parse((APP.parent/'secureagent_ui'/'contracts.py').read_text())
    paths={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('/api/')}
    # Phase 8 adds one fixed prefix; its id segment must match a 32-hex pattern.
    assert paths=={'/api/v1/chat','/api/v1/rag/chat','/api/v1/agent/chat','/api/v1/agent/approvals/'}


def test_all_concrete_providers_and_factory_are_isolated():
    forbidden = ("app.llm.ollama_client", "app.llm.foundry_client", "app.llm.factory")
    for path, tree in _modules().items():
        if path == COMPOSITION_ROOT or APP / "llm" in path.parents:
            continue
        assert not any(any(name == f or name.startswith(f + ".") for f in forbidden)
                       for name in _imported_names(tree)), str(path)
