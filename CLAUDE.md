# faultline

A Claude agent harness for diagnosing and fixing network faults safely, plus an eval bench that measures it. See `README.md` for the goal and `docs/PLAN.md` for the build plan.

## Ground rules

- Python 3.11+, `src/` layout. Shared contracts live in `src/faultline/types.py`. Build against them; change them only on `main`.
- Calls go through the official `anthropic` SDK (pinned `<2`). Never add other LLM SDKs or agent frameworks; owning the loop is the point of the project.
- **History is append-only.** Never edit or remove a message once it is in the conversation. Truncate tool output before it enters history; compact by starting a fresh conversation. Editing history breaks preserved thinking on the 5.5 models.
- Every test must run without Docker and without an API key. Use `FakeLLMClient` and `ScriptedLab`. Anything that needs Docker or a real key is marked `@pytest.mark.live`.
- Run `python -m pytest` and `python -m ruff check src tests` before every commit.

## Git

- One branch per feature (`feat/<name>`), small focused commits, imperative subject lines.
- Never add AI attribution: no `Co-Authored-By` trailers for Claude or any AI, no "Generated with" lines, in commits or PRs.
