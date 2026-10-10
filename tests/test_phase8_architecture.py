"""Phase 8 architecture guards for LangGraph orchestration and the MCP tool layer.

    API -> SecureLLMService facade -> LangGraph runtime -> nodes
    nodes -> ToolGateway (policy) -> MCP gateway -> MCP client -> in-process MCP servers -> registry
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app'


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _imports(path):
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f'{node.module}.{alias.name}' for alias in node.names)
    return names


def _hits(path, forbidden):
    return sorted(name for name in _imports(path) for f in forbidden if name == f or name.startswith(f + '.'))


def _files(*parts):
    return sorted((APP.joinpath(*parts)).rglob('*.py'))


def test_api_reaches_only_the_graph_service_facade():
    forbidden = ('app.graph', 'app.mcp', 'app.tools', 'app.agent.tool_gateway', 'app.agent.tool_registry',
                 'langgraph', 'mcp', 'mcp_types')
    for path in _files('api'):
        assert _hits(path, forbidden) == [], path.name
    source = (APP / 'api' / 'routes.py').read_text()
    for internal in ('runtime', '_runtime', 'tool_gateway', 'graph', 'checkpointer'):
        assert internal not in source


def test_service_request_methods_delegate_to_the_langgraph_runtime():
    tree = _tree(APP / 'security' / 'secure_llm.py')
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    expected = {'chat': 'run', 'rag_chat': 'run', 'agent_chat': 'run', 'decide_approval': 'resume'}
    for method, call in expected.items():
        calls = [n for n in ast.walk(methods[method]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == call and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == '_runtime']
        assert calls, f'{method} must run the compiled graph'


def test_ui_cannot_reach_langgraph_or_mcp():
    forbidden = ('app.graph', 'app.mcp', 'langgraph', 'mcp', 'mcp_types', 'langchain_core', 'langsmith')
    for path in [*_files('ui'), *sorted((ROOT / 'secureagent_ui').glob('*.py'))]:
        assert _hits(path, forbidden) == [], path.name


def test_llm_providers_cannot_reach_mcp_or_graph():
    for path in _files('llm'):
        assert _hits(path, ('app.mcp', 'mcp', 'mcp_types', 'app.graph', 'langgraph', 'app.tools')) == [], path.name


def test_graph_and_planner_cannot_reach_concrete_tools_mcp_or_providers():
    forbidden = ('app.tools', 'app.agent.tool_registry', 'app.mcp.servers', 'app.mcp.host', 'app.mcp.client',
                 'app.mcp.gateway', 'mcp', 'mcp_types', 'app.llm.ollama_client', 'app.llm.foundry_client',
                 'app.llm.factory', 'httpx', 'requests', 'socket', 'subprocess', 'os', 'sqlite3', 'pickle')
    for path in [*_files('graph'), APP / 'agent' / 'planner.py']:
        assert _hits(path, forbidden) == [], path.name


def test_graph_reaches_tools_only_through_the_policy_gateway():
    tree = _tree(APP / 'graph' / 'nodes.py')
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert {'authorize_and_validate', 'execute_raw', 'guard_result', 'discover'} <= attributes
    assert 'dispatch' not in attributes and 'call_tool' not in attributes and 'list_tools' not in attributes


def test_mcp_servers_cannot_reach_models_policy_bypasses_or_os():
    forbidden = ('app.llm', 'app.security', 'app.graph', 'app.agent.authorization', 'app.agent.approval',
                 'app.agent.tool_gateway', 'app.mcp.gateway', 'app.mcp.client', 'httpx', 'requests', 'socket',
                 'subprocess', 'os', 'sqlite3', 'importlib', 'pickle', 'shutil', 'pathlib')
    for path in _files('mcp', 'servers'):
        assert _hits(path, forbidden) == [], path.name


def test_only_the_policy_gateway_reaches_the_mcp_gateway():
    for path in APP.rglob('*.py'):
        if _hits(path, ('app.mcp.gateway',)):
            assert path == APP / 'agent' / 'tool_gateway.py', path


def test_mcp_client_host_and_servers_have_single_owners():
    owners = {'app.mcp.client': {APP / 'mcp' / 'gateway.py'}, 'app.mcp.host': {APP / 'mcp' / 'gateway.py'},
              'app.mcp.servers': {APP / 'mcp' / 'host.py'}}
    for path in APP.rglob('*.py'):
        for module, allowed in owners.items():
            if path.parent != APP / 'mcp' / 'servers' and _hits(path, (module,)):
                assert path in allowed, f'{path.relative_to(APP)} imports {module}'


def test_only_the_mcp_client_uses_the_sdk_client_and_never_remote_transports():
    remote = ('mcp.client.stdio', 'mcp.client.streamable_http', 'mcp.client.sse', 'mcp.server.stdio',
              'mcp.server.sse', 'mcp.server.streamable_http', 'mcp.server.streamable_http_manager',
              'mcp.StdioServerParameters', 'mcp.stdio_client', 'uvicorn', 'starlette')
    for path in [*_files('mcp'), *_files('graph'), *_files('agent')]:
        assert _hits(path, remote) == [], path.name
        if path != APP / 'mcp' / 'client.py':
            assert _hits(path, ('mcp.Client', 'mcp.client', 'mcp.ClientSession')) == [], path.name
    tree = _tree(APP / 'mcp' / 'client.py')
    clients = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Client']
    assert len(clients) == 1
    keywords = {k.arg: k.value for k in clients[0].keywords}
    assert isinstance(keywords['mode'], ast.Constant) and keywords['mode'].value == 'legacy'
    # No callbacks: an MCP server cannot request LLM sampling, user elicitation or local roots.
    assert not {'sampling_callback', 'elicitation_callback', 'list_roots_callback'} & set(keywords)


def test_rag_cannot_reach_mcp_or_graph():
    for path in _files('rag'):
        assert _hits(path, ('app.mcp', 'mcp', 'mcp_types', 'app.graph', 'langgraph')) == [], path.name


def test_no_dynamic_code_execution_in_orchestration_or_tools():
    for path in [*_files('graph'), *_files('mcp'), *_files('agent'), *_files('tools')]:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ('eval', 'exec', 'compile', '__import__', 'open'), path.name
        assert _hits(path, ('importlib',)) == [], path.name


def test_no_cloud_graph_services_or_persistent_checkpointers():
    forbidden = ('langgraph_sdk', 'langgraph.pregel.remote', 'langgraph.checkpoint.sqlite',
                 'langgraph.checkpoint.postgres', 'langgraph.store', 'langsmith.client', 'langsmith.Client',
                 'langchain', 'langchain_core', 'crewai', 'autogen', 'llama_index')
    for path in APP.rglob('*.py'):
        assert _hits(path, forbidden) == [], path.name
    runtime = (APP / 'graph' / 'runtime.py').read_text()
    assert 'InMemorySaver()' in runtime and 'tracing_context(enabled=False)' in runtime


def test_dependencies_are_pinned_and_minimal():
    requirements = (ROOT / 'requirements.txt').read_text().splitlines()
    pinned = {line.split('==')[0] for line in requirements if '==' in line and not line.startswith('#')}
    assert {'langgraph', 'mcp'} <= pinned
    for banned in ('langchain', 'langchain-core', 'crewai', 'autogen', 'llama-index', 'langsmith', 'openai'):
        assert banned not in pinned
