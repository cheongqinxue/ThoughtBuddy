# `history/` — message archive review

Paginated read and BM25 search over append-only `message_log`.

| File | Role |
|---|---|
| [`tools.py`](tools.py) | `history_read`, `history_search` |

Store: [`store/messages.py`](../store/messages.py) · Text helpers: [`context/message_text.py`](../context/message_text.py)

Tests: `tests/test_history.py`
