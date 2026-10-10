"""Offline, bounded repository hygiene/secret check. Never prints matched values.

Scans tracked files and nonignored untracked files; tracked private artifacts
remain visible even when .gitignore would exclude them. This is a heuristic,
not a replacement for dedicated secret scanning or human data review.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----'),
    'provider_key': re.compile(r'\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b'),
    'credential_assignment': re.compile(r'''(?i)\b(?:api[_-]?key|password|client[_-]?secret|access[_-]?token|secret[_-]?key)\s*[=:]\s*["']?([A-Za-z0-9_+/=-]{12,})'''),
    'authorization_value': re.compile(r'(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9_+/=-]{20,}'),
    'credential_url': re.compile(r'''(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@"']+:[^\s/@"']+@'''),
}
HOME_PATH = re.compile(r'/(?:Users|home)/[A-Za-z0-9._-]+/')
PRIVATE_SUFFIXES = ('.jsonl','.log','.db','.sqlite','.sqlite3','.db-journal','.db-wal','.db-shm',
    '.sqlite-journal','.sqlite-wal','.sqlite-shm','.sqlite3-journal','.sqlite3-wal','.sqlite3-shm',
    '.gguf','.safetensors','.onnx','.pt','.pth','.pem','.key','.tmp','.temp','.bak','.swp','.swo','.pyc')


def private_artifact(path):
    path = Path(path)
    return ((path.name.startswith('.env') and path.name != '.env.example')
        or path.name.startswith('secrets') and '.streamlit' in path.parts
        or path.name.endswith(PRIVATE_SUFFIXES)
        or any(p in {'.venv','venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','logs','vectorstore','chroma'} for p in path.parts))


def scan_text(text):
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for category, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append({'line': line_number, 'category': category})
    return findings


def repository_files(root):
    result = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                            cwd=root, check=True, capture_output=True)
    return sorted(set(p.decode('utf-8') for p in result.stdout.split(b'\0') if p))


def check(root=ROOT):
    findings = []
    fixture_paths = []
    candidates = repository_files(root)
    for name in candidates:
        path = root / name
        if private_artifact(name):
            findings.append({'file':name, 'category':'private_or_generated_artifact'})
            continue
        if path.is_symlink():
            findings.append({'file':name, 'category':'symlink_not_scanned'})
            continue
        if not path.exists():  # A deleted tracked file is absent from the release.
            continue
        if path.stat().st_size > 2_000_000:
            findings.append({'file':name, 'category':'oversized_not_scanned'})
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (UnicodeError,OSError):
            findings.append({'file':name, 'category':'unreadable_or_binary'})
            continue
        findings.extend({'file':name, **item} for item in scan_text(text))
        if HOME_PATH.search(text):
            if Path(name).parts[0] == 'tests':
                fixture_paths.append(name)
            else:
                findings.append({'file':name, 'category':'developer_home_path'})
    return {'result':'FAIL' if findings else 'PASS', 'files_checked':len(candidates),
            'findings':findings, 'negative_test_path_fixtures':fixture_paths}


if __name__ == '__main__':
    report = check()
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['findings'] else 0)
