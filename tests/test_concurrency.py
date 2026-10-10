"""Concurrent requests must never share token mappings.

Both requests receive the same token (<PERSON_1>, <EMAIL_1>), so any shared
or crossed vault would re-identify one person's response with the other's
data. A barrier holds every request inside the LLM call until all of them
are in flight, guaranteeing that their vaults are alive at the same time.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor


class BarrierLLM:
    """Waits until `parties` requests are inside generate(), then echoes the prompt."""

    model_name = "qwen2.5:7b"

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=20)
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        with self._lock:
            self.prompts.append(prompt)
        self.barrier.wait()  # BrokenBarrierError -> test fails if requests did not overlap
        return f"Noted: {prompt}"


ALLOW = ["PERSON", "EMAIL_ADDRESS"]


def _post_all(client, messages: list[str]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(messages)) as pool:
        futures = [
            pool.submit(client.post, "/api/v1/chat",
                        json={"message": m, "allowed_reidentification_types": ALLOW})
            for m in messages
        ]
        responses = [f.result() for f in futures]
    assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]
    return [r.json() for r in responses]


def test_two_simultaneous_requests_do_not_cross_mappings(make_harness):
    llm = BarrierLLM(parties=2)
    h = make_harness(llm=llm)
    david = "My name is David Chen and my email is david.chen@example.com."
    maria = "My name is Maria Garcia and my email is maria.garcia@example.com."

    a, b = _post_all(h.client, [david, maria])

    # Both prompts used identical tokens, so only per-request vaults can tell them apart.
    assert sorted(llm.prompts) == ["My name is <PERSON_1> and my email is <EMAIL_1>."] * 2
    assert "David Chen" in a["response"] and "david.chen@example.com" in a["response"]
    assert "Maria Garcia" not in a["response"] and "maria.garcia" not in a["response"]
    assert "Maria Garcia" in b["response"] and "maria.garcia@example.com" in b["response"]
    assert "David Chen" not in b["response"] and "david.chen" not in b["response"]
    assert a["request_id"] != b["request_id"]


def test_many_simultaneous_requests_stay_isolated(make_harness):
    n = 12
    llm = BarrierLLM(parties=n)
    h = make_harness(llm=llm)
    messages = [f"Please email employee{i:02d}@example.com about the update." for i in range(n)]

    results = _post_all(h.client, messages)

    for i, body in enumerate(results):
        assert f"employee{i:02d}@example.com" in body["response"]
        others = [f"employee{j:02d}@" for j in range(n) if j != i]
        assert not any(o in body["response"] for o in others)
    assert set(llm.prompts) == {"Please email <EMAIL_1> about the update."}
