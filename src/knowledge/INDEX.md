# `knowledge/` — swappable knowledge backend

| File | Role |
|---|---|
| [`backend.py`](backend.py) | `KnowledgeBackend` protocol |
| [`chroma.py`](chroma.py) | Chroma v1 implementation (internal) |
| [`tools.py`](tools.py) | `knowledge_add`, `knowledge_edit`, `knowledge_search` |

Evolved from [`memory/`](../memory/). Tests: `tests/test_knowledge.py`
