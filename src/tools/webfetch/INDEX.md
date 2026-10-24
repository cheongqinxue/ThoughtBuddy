# `tools/webfetch/` — webfetch agent tool

Agent-facing tool name: **`webfetch`**. Playwright rendering + trafilatura extraction; SSRF guard in `url_safety.py`.

## Layout

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | `create_tools()` — registers `webfetch` |
| [`fetcher.py`](fetcher.py) | `PageFetcher` — headless Chromium load + retry |
| [`format.py`](format.py) | HTML → readable text for the LLM |
| [`schemas.py`](schemas.py) | `WebFetchInput` args schema |
| [`url_safety.py`](url_safety.py) | `validate_fetch_url()` — SSRF mitigation |
