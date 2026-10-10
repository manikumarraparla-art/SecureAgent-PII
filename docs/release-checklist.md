# v1.0.0 release checklist

The code advertises 1.0.0 for reproducible release-candidate verification.
It is not a published release until the owner finishes the unchecked gates.
Record evidence and date in [Phase 9 report](phase-9-report.md); do not treat a
successful health response or a green historical CI run as current evidence.

## Local review gates

- [x] Confirm starting branch and clean Git status; run the 719-test baseline.
- [x] Run the full final suite: `python -m pytest -v`; record exact count/warnings.
- [x] Run `python scripts/check_repository.py`; review findings without printing values.
- [x] Run `python -m pip check` and `python -m compileall -q app secureagent_ui scripts tests`.
- [x] Run `git diff --check` and inspect `git diff --stat` / `git status --short --branch`.
- [x] Review tracked and candidate files for credentials, raw real PII, runtime data,
      vector stores, logs, caches, local paths and model artifacts; retain synthetic fixtures.
- [x] Verify ignore rules with `git check-ignore`; do not assume ignored files were never tracked.
- [x] Run `python -m scripts.local_ui_startup_smoke` for fresh API/UI + actual WebSocket chat.
- [x] Run `python -m scripts.local_release_smoke --provider ollama` and `--provider foundry`.
- [x] Run `python scripts/local_ui_smoke.py` and `python -m scripts.local_provider_smoke --provider foundry`.
- [x] Review real MCP authorization, scope denial, approval/denial and tool failures;
      check privacy at model boundaries and in audit/traces.
- [x] Review README, architecture, five-minute demo, dependency decisions, threat
      model, production gaps and exact startup commands against the implementation.
- [x] Confirm no new cloud, model, database or orchestration component; no validation
      weakening or routing around the security service.

## Owner-only publication gates (intentionally pending)

- [ ] Review/approve the final working-tree changes and Phase 9 evidence.
- [ ] Run independent secret/history/provenance review appropriate for public release;
      the bundled scanner is heuristic and scans the working tree, not all Git history.
- [ ] Create the final reviewed commit and merge using the repository's normal workflow.
- [ ] Push only after review and confirm **this revision's** full GitHub Actions job passes.
      CI must stay generation-model-free; it installs spaCy NER data, not Ollama weights.
- [ ] Confirm the published README/version matches the approved commit.
- [ ] Optionally create and push the `v1.0.0` tag and release notes after all gates pass.

Phase 9 automation must not commit, merge, push or create a tag. Do not mark the
hosted CI or publication gates complete based on local tests.
