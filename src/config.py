"""Application configuration loaded from config.yaml and .env secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.yaml")

# Map nested YAML paths to Settings field names.
_YAML_FIELD_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("llm", "openai_base_url"), "openai_base_url"),
    (("llm", "agent_model"), "agent_model"),
    (("llm", "tokenizer_model"), "llm_tokenizer_model"),
    (("llm", "temperature"), "llm_temperature"),
    (("llm", "max_tokens"), "llm_max_tokens"),
    (("llm", "context_window_tokens"), "llm_context_window_tokens"),
    (("logging", "level"), "log_level"),
    (("logging", "retry", "log_each_attempt"), "retry_log_each_attempt"),
    (("logging", "retry", "max_attempts"), "retry_max_attempts"),
    (("logging", "retry", "min_wait_seconds"), "retry_min_wait_seconds"),
    (("logging", "retry", "max_wait_seconds"), "retry_max_wait_seconds"),
    (("llm", "max_retries"), "llm_max_retries"),
    (("llm", "enable_thinking"), "llm_enable_thinking"),
    (("runtime", "db_path"), "thoughtbuddy_db_path"),
    (("runtime", "lock_path"), "thoughtbuddy_lock_path"),
    (("runtime", "heartbeat_interval_seconds"), "heartbeat_interval_seconds"),
    (("runtime", "stale_task_seconds"), "stale_task_seconds"),
    (("runtime", "default_thread_id"), "default_thread_id"),
    (("runtime", "upcoming_alerts_window_seconds"), "upcoming_alerts_window_seconds"),
    (("monitor", "max_concurrent"), "monitor_max_concurrent"),
    (("monitor", "default_timeout_seconds"), "monitor_default_timeout_seconds"),
    (("monitor", "min_expected_seconds"), "monitor_min_expected_seconds"),
    (("monitor", "min_expected_heartbeats"), "monitor_min_expected_heartbeats"),
    (("monitor", "log_tail_bytes"), "monitor_log_tail_bytes"),
    (("context", "retain_messages"), "context_retain_messages"),
    (("context", "compact_token_threshold"), "context_compact_token_threshold"),
    (("context", "compact_token_ratio"), "context_compact_token_ratio"),
    (("websearch", "provider"), "websearch_provider"),
    (("webfetch", "timeout_seconds"), "webfetch_timeout_seconds"),
    (("webfetch", "network_idle_timeout_ms"), "webfetch_network_idle_timeout_ms"),
    (("webfetch", "post_load_wait_ms"), "webfetch_post_load_wait_ms"),
    (("webfetch", "max_chars"), "webfetch_max_chars"),
    (("memory", "store_path"), "memory_store_path"),
    (("memory", "collection_name"), "memory_collection_name"),
    (("memory", "embedding_model"), "memory_embedding_model"),
    (("memory", "embedding_device"), "memory_embedding_device"),
    (("memory", "recall_multiplier"), "memory_recall_multiplier"),
    (("scratchpad", "directory"), "scratchpad_directory"),
    (("experiment", "active"), "active_experiment"),
    (("experiment", "chess", "engine_path"), "chess_engine_path"),
    (("experiment", "chess", "agent_color"), "chess_agent_color"),
    (("experiment", "chess", "agent_generation"), "chess_agent_generation"),
    (("experiment", "chess", "skill_level"), "chess_skill_level"),
    (("experiment", "chess", "move_time_seconds"), "chess_move_time_seconds"),
    (("experiment", "chess", "engine_timeout_seconds"), "chess_engine_timeout_seconds"),
    (("observability", "langfuse", "enabled"), "langfuse_enabled"),
    (("observability", "langfuse", "base_url"), "langfuse_base_url"),
    (("observability", "langfuse", "log_min_level"), "langfuse_log_min_level"),
)

# Secrets loaded exclusively from .env.
_SECRET_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("openai_api_key", "OPENAI_API_KEY"),
    ("langsearch_api_key", "LANGSEARCH_API_KEY"),
    ("event_webhook_secret", "EVENT_WEBHOOK_SECRET"),
)


class Settings(BaseModel):
    """Runtime settings for the ThoughtBuddy agent."""

    openai_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible API base URL.",
    )
    openai_api_key: str = Field(
        default="EMPTY",
        description="API key for the inference server.",
    )
    agent_model: str = Field(
        default="Qwen/Qwen3-8B-AWQ",
        description="Model name served by the inference server.",
    )
    llm_tokenizer_model: str = Field(
        default="",
        description="Hugging Face tokenizer hub id; defaults to agent_model when empty.",
    )
    llm_temperature: float = Field(
        default=0.2,
        description="Sampling temperature for the agent chat model.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Max completion tokens per LLM call (prevents runaway generation).",
    )
    llm_context_window_tokens: int = Field(
        default=32768,
        ge=1024,
        description="Model context window size used for wake budget checks.",
    )
    log_level: str = Field(
        default="INFO",
        description="Application log level.",
    )
    retry_log_each_attempt: bool = Field(
        default=False,
        description="Log each API retry attempt at DEBUG. Final failure always logs at ERROR.",
    )
    retry_max_attempts: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum attempts for shared ApiRetry helpers (first call included).",
    )
    retry_min_wait_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum backoff between ApiRetry attempts.",
    )
    retry_max_wait_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Maximum backoff between ApiRetry attempts.",
    )
    llm_enable_thinking: bool = Field(
        default=False,
        description="Qwen3 chat-template thinking mode; keep false for tool-calling harness.",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Max retries for OpenAI-compatible LLM requests via LangChain.",
    )
    websearch_provider: str = Field(
        default="langsearch",
        description="Web search backend id (see tools/websearch/providers/).",
    )
    langsearch_api_key: str = Field(
        default="",
        description="API key for LangSearch when websearch.provider is langsearch.",
    )
    thoughtbuddy_db_path: str = Field(
        default="data/thoughtbuddy.db",
        description="SQLite database path for alerts, tasks, and message archive.",
    )
    thoughtbuddy_lock_path: str = Field(
        default="data/thoughtbuddy.lock",
        description="PID lock file path to enforce a single runtime instance.",
    )
    heartbeat_interval_seconds: int = Field(
        default=60,
        description="Hidden heartbeat poll interval in seconds.",
    )
    stale_task_seconds: int = Field(
        default=1800,
        description="Requeue running tasks older than this many seconds.",
    )
    default_thread_id: str = Field(
        default="thoughtbuddy-main",
        description="Default LangGraph checkpoint thread id.",
    )
    upcoming_alerts_window_seconds: int = Field(
        default=3600,
        description="Lookahead window for upcoming alerts in wake context.",
    )
    event_webhook_secret: str = Field(
        default="",
        description="Shared secret for POST /thoughtbuddy/tasks/event (from .env).",
    )
    monitor_max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent monitored subprocesses.",
    )
    monitor_default_timeout_seconds: int = Field(
        default=600,
        ge=1,
        description="Default monitored subprocess timeout (10 minutes).",
    )
    monitor_min_expected_seconds: int = Field(
        default=60,
        ge=1,
        description="Documented minimum expected monitor duration in seconds.",
    )
    monitor_min_expected_heartbeats: int = Field(
        default=5,
        ge=1,
        description="Minimum expected monitor duration measured in heartbeat ticks.",
    )
    monitor_log_tail_bytes: int = Field(
        default=8192,
        ge=256,
        description="Bytes of stdout/stderr tail included on monitor completion events.",
    )
    context_retain_messages: int = Field(
        default=20,
        description="Verbatim messages kept before each compaction boundary.",
    )
    context_compact_token_threshold: int = Field(
        default=0,
        ge=0,
        description=(
            "Compact archived context when estimate exceeds this token count. "
            "When 0, threshold is compact_token_ratio × max_prompt_tokens."
        ),
    )
    context_compact_token_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of max_prompt_tokens that triggers archive compaction when "
            "compact_token_threshold is 0. Values ≤ 0 fall back to 0.9."
        ),
    )
    webfetch_timeout_seconds: int = Field(
        default=30,
        description="Maximum seconds to wait for a webfetch page load.",
    )
    webfetch_network_idle_timeout_ms: int = Field(
        default=5_000,
        description="Extra wait for network idle after load; 0 skips.",
    )
    webfetch_post_load_wait_ms: int = Field(
        default=1_000,
        description="Fixed delay after load for lazy JavaScript rendering.",
    )
    webfetch_max_chars: int = Field(
        default=40_000,
        description="Default maximum extracted characters returned by webfetch.",
    )
    memory_store_path: str = Field(
        default="data/memories/chroma",
        description="On-disk path for the ChromaDB memory store.",
    )
    memory_collection_name: str = Field(
        default="thoughtbuddy_memories",
        description="Chroma collection name for agent memories.",
    )
    memory_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers model for memory embeddings.",
    )
    memory_embedding_device: str = Field(
        default="",
        description="Torch device for memory embeddings; empty auto-selects CUDA/CPU.",
    )
    memory_recall_multiplier: int = Field(
        default=4,
        ge=1,
        description="Default recall pool size multiplier before optional reranking.",
    )
    scratchpad_directory: str = Field(
        default=".data/scratchpad",
        description="Directory for persistent scratchpad markdown files.",
    )
    active_experiment: str = Field(
        default="",
        description="Active experiment id (e.g. chess). Empty disables experiment tools.",
    )
    chess_engine_path: str = Field(
        default=".data/stockfish/engine",
        description="Stockfish executable path or command name for the chess experiment.",
    )
    chess_agent_color: str = Field(
        default="white",
        description="Side the agent plays in chess (white or black).",
    )
    chess_agent_generation: int = Field(
        default=0,
        ge=0,
        description="Agent generation label stored on chess sessions and replays.",
    )
    chess_skill_level: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Stockfish Skill Level (0 weakest, 20 strongest).",
    )
    chess_move_time_seconds: float = Field(
        default=0.1,
        ge=0.01,
        description="Seconds Stockfish may think per opponent move.",
    )
    chess_engine_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Timeout for starting Stockfish UCI sessions.",
    )
    langfuse_enabled: bool = Field(
        default=False,
        description="Enable Langfuse tracing when LANGFUSE_* credentials are configured.",
    )
    langfuse_base_url: str = Field(
        default="",
        description="Optional Langfuse API base URL override (defaults to SDK/env).",
    )
    langfuse_log_min_level: str = Field(
        default="WARNING",
        description="Minimum Python log level forwarded to Langfuse spans.",
    )

    @classmethod
    def load(
        cls,
        *,
        config_path: str | Path | None = None,
        env_file: str | Path | None = ".env",
    ) -> Self:
        """Load settings from config.yaml with secrets from .env.

        Non-sensitive values come from ``config.yaml`` (or pydantic defaults when
        the file or a key is missing). API keys are read only from ``.env``.

        Args:
            config_path: Path to the YAML config file. Defaults to ``config.yaml``
                in the current working directory.
            env_file: Path to the dotenv secrets file. Defaults to ``.env``.

        Returns:
            Parsed settings instance.
        """
        yaml_values = _load_yaml_values(config_path or DEFAULT_CONFIG_PATH)
        secrets = _load_secrets(env_file)
        return cls(**{**yaml_values, **secrets})


def _load_yaml_values(config_path: Path) -> dict[str, Any]:
    """Extract Settings fields from nested YAML content.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Flat mapping of Settings field names to values.
    """
    if not config_path.exists():
        return {}

    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        msg = f"Expected mapping at root of {config_path}"
        raise ValueError(msg)

    values: dict[str, Any] = {}
    for path, field_name in _YAML_FIELD_PATHS:
        node: Any = data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
        else:
            values[field_name] = node
    return values


def _load_secrets(env_file: str | Path | None) -> dict[str, str]:
    """Load secret values from environment variables.

    Args:
        env_file: Optional dotenv file path.

    Returns:
        Mapping of Settings secret field names to values present in the environment.
    """
    if env_file is not None:
        load_dotenv(env_file)

    secrets: dict[str, str] = {}
    for field_name, env_var in _SECRET_ENV_VARS:
        value = os.getenv(env_var)
        if value is not None:
            secrets[field_name] = value
    return secrets
