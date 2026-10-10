"""Cold-process regression: other tests must not pre-import the backend package."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_launcher_runs_without_preimported_app_package():
    code = '''
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
assert "app" not in sys.modules
ui = AppTest.from_file(Path("app/ui/app.py").resolve()).run(timeout=20)
assert not ui.exception
assert ui.title[0].value == "SecureAgent-PII"
assert ui.selectbox(key="mode").value == "Secure Chat"
assert len(ui.chat_input) == 1
assert "app" not in sys.modules, "UI must load without importing the backend package"
print("cold-start PASS")
'''
    environment = {**os.environ, 'UI_BACKEND_URL': 'http://127.0.0.1:8000', 'UI_CONNECT_TIMEOUT': '1'}
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'cold-start PASS' in result.stdout


def test_ui_and_backend_package_initializers_and_compatibility_exports():
    for package in ('app', 'app/ui', 'secureagent_ui'):
        assert (ROOT / package / '__init__.py').is_file()
    from app.ui.api_client import APIClient as legacy_client
    from secureagent_ui.api_client import APIClient
    assert legacy_client is APIClient
