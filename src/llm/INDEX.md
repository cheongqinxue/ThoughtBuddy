# `llm/` — inference client and token counting

Model identity comes from `config.yaml` → `llm.agent_model`. Tokenizer hub id defaults to the same value unless `llm.tokenizer_model` is set.

## Layout

| Module | Role |
|---|---|
| [`factory.py`](factory.py) | `create_chat_model()` → `ChatOpenAI` pointed at vLLM |
| [`chat_template.py`](chat_template.py) | Qwen3 `enable_thinking` kwargs for inference + token counting |
| [`profile.py`](profile.py) | `ModelProfile`, `resolve_model_profile()` |
| [`tokens.py`](tokens.py) | `count_messages_tokens()` — public counting API |
| [`tokenizers/huggingface.py`](tokenizers/huggingface.py) | Default: HF `apply_chat_template` + encode |
| [`tokenizers/vllm.py`](tokenizers/vllm.py) | Optional: `/v1/chat/completions/render` (integration verify) |
| [`tokenizers/messages.py`](tokenizers/messages.py) | LangChain → OpenAI dict (same path as `ChatOpenAI`) |

## Deployment today

- **Inference:** vLLM (`scripts/start_vllm_agent.sh`) — separate conda env, OpenAI API on `:8000/v1`
- **Harness:** `create_chat_model()` — HTTP client only; no local weights
- **Token counting:** HuggingFace tokenizer loaded in-process (cached); matches vLLM when hub ids align

## Tests

| File | Covers |
|---|---|
| `tests/test_llm_tokens.py` | HF counting, profile resolution |
| `tests/integration/test_vllm_tokens.py` | HF vs vLLM render parity (integration) |
