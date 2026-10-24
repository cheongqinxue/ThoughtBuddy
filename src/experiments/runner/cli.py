"""CLI entry points for root run.sh."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from config import Settings
from experiments.runner.apply import apply_to_app_config
from experiments.runner.config import ExperimentRunConfig, shell_env
from experiments.runner.entrypoint import SetupError, run_analyze, run_preflight, start_hints
from runtime.failure_report import clear_experiment_failure_report


def _repo_root() -> Path:
    return Path.cwd()


def _load_config(path: str) -> ExperimentRunConfig:
    return ExperimentRunConfig.load(Path(path), repo_root=_repo_root())


def cmd_apply(args: argparse.Namespace) -> int:
    """Sync experiment YAML into config.yaml."""
    run_config = _load_config(args.config)
    app_path = apply_to_app_config(run_config, repo_root=_repo_root())
    print(f"Applied {run_config.config_path} → {app_path}")
    print(f"  experiment: {run_config.experiment}")
    print(f"  entrypoint: {run_config.entrypoint}")
    print(f"  llm.profile: {run_config.llm_profile}")
    return 0


def cmd_write_env(args: argparse.Namespace) -> int:
    """Write shell-friendly env file for run.sh."""
    run_config = _load_config(args.config)
    paths = run_config.resolved_paths(_repo_root())
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    paths["pid_dir"].mkdir(parents=True, exist_ok=True)

    env = shell_env(run_config, repo_root=_repo_root())
    out_path = Path(args.output) if args.output else paths["run_dir"] / "run.env"
    lines = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Run experiment preflight checks."""
    run_config = _load_config(args.config)
    try:
        run_preflight(run_config, repo_root=_repo_root())
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_langfuse_enabled(_args: argparse.Namespace) -> int:
    """Print whether Langfuse is enabled in app config (true/false)."""
    enabled = Settings.load().langfuse_enabled
    print("true" if enabled else "false")
    return 0


def cmd_clear_failure(_args: argparse.Namespace) -> int:
    """Remove stale experiment failure report."""
    clear_experiment_failure_report(Settings.load())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Print experiment analysis summary."""
    run_config = _load_config(args.config)
    run_analyze(run_config, repo_root=_repo_root())
    print()
    print(f"Tip: rg 'wake=' {run_config.resolved_paths(_repo_root())['thoughtbuddy_log']} | tail -20")
    return 0


def cmd_start_hints(args: argparse.Namespace) -> int:
    """Print experiment-specific startup hints for run.sh."""
    run_config = _load_config(args.config)
    for line in start_hints(run_config, tb_port=int(args.tb_port)):
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run an experiment runner subcommand.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="ThoughtBuddy experiment runner")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("apply", help="Sync experiment YAML into config.yaml")
    write_env = sub.add_parser("write-env", help="Write run.env for run.sh")
    write_env.add_argument(
        "--output",
        default=None,
        help="Output env file path (default: <run_dir>/run.env)",
    )
    sub.add_parser("preflight", help="Run experiment setup checks")
    sub.add_parser("langfuse-enabled", help="Print true/false from app config")
    sub.add_parser("clear-failure", help="Remove stale failure.json")
    sub.add_parser("analyze", help="Print experiment summary")
    hints = sub.add_parser("start-hints", help="Print post-start banner lines")
    hints.add_argument("--tb-port", required=True, help="ThoughtBuddy HTTP port")

    args = parser.parse_args(argv)
    handlers = {
        "apply": cmd_apply,
        "write-env": cmd_write_env,
        "preflight": cmd_preflight,
        "langfuse-enabled": cmd_langfuse_enabled,
        "clear-failure": cmd_clear_failure,
        "analyze": cmd_analyze,
        "start-hints": cmd_start_hints,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
