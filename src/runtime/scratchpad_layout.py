"""Seed the scratchpad directory from committed templates."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_DIR = Path(".data-template/scratchpad")
SCRATCHPAD_FILENAME = "SCRATCHPAD.md"
PROGRESS_FILENAME = "PROGRESS.md"
COMPACT_FILENAME = "_compact.md"
SCRATCHPAD_DIR_FILES = (
    "_goals.md",
    "_system.md",
    "_compact.md",
    SCRATCHPAD_FILENAME,
    PROGRESS_FILENAME,
)


def ensure_scratchpad_layout(
    directory: str | Path,
    *,
    template_dir: Path | None = None,
) -> None:
    """Ensure scratchpad directory exists with default markdown files.

    Owner-controlled ``_*.md`` files plus agent-editable ``SCRATCHPAD.md`` and
    ``PROGRESS.md`` live together under ``.data/scratchpad/``.

    Args:
        directory: Target scratchpad directory.
        template_dir: Source templates when files are missing.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    source = template_dir or DEFAULT_TEMPLATE_DIR
    for filename in SCRATCHPAD_DIR_FILES:
        dest = target / filename
        if dest.exists():
            continue
        template_path = source / filename
        if template_path.is_file():
            shutil.copyfile(template_path, dest)
            logger.info("Seeded scratchpad directory file %s", dest)
        else:
            dest.write_text("", encoding="utf-8")
