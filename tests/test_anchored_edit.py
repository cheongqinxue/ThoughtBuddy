"""Tests for anchored file edit utilities and agent tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from runtime.scratchpad_layout import PROGRESS_FILENAME, SCRATCHPAD_FILENAME, ensure_scratchpad_layout
from tools.progress import create_tools as create_progress_tools
from tools.scratchpad import create_tools as create_scratchpad_tools
from utils.anchored_edit import BOF_MARKER, EOF_MARKER, AnchoredEditError, apply_edit, edit_file, read_file


@pytest.fixture
def scratchpad_dir(tmp_path: Path) -> Path:
    """Return an isolated scratchpad directory seeded from templates."""
    directory = tmp_path / "scratchpad"
    template_dir = Path(".data-template/scratchpad")
    ensure_scratchpad_layout(directory, template_dir=template_dir)
    return directory


@pytest.fixture
def scratchpad_settings(scratchpad_dir: Path) -> Settings:
    """Settings pointing at a temporary scratchpad directory."""
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test-key",
        agent_model="test-model",
        langsearch_api_key="test-langsearch-key",
        scratchpad_directory=str(scratchpad_dir),
    )


class TestAnchoredEdit:
    """Anchored file edit tests."""

    def test_read_returns_content_as_is(self, scratchpad_dir: Path) -> None:
        """Return file contents without trimming."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("  note one  \n", encoding="utf-8")

        assert read_file(path) == "  note one  \n"

    def test_read_missing_file_returns_empty(self, scratchpad_dir: Path) -> None:
        """Return an empty string when the file is missing."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.unlink()

        assert read_file(path) == ""

    def test_replace_single_line(self, scratchpad_dir: Path) -> None:
        """Replace one line between anchors."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        updated = edit_file(path, pre="alpha", post="gamma", original="beta", new="bravo")

        assert updated == "alpha\nbravo\ngamma\n"
        assert read_file(path) == updated

    def test_replace_multiline_region(self, scratchpad_dir: Path) -> None:
        """Replace a multi-line original block."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("start\nold one\nold two\nend\n", encoding="utf-8")

        updated = edit_file(
            path,
            pre="start",
            post="end",
            original="old one\nold two",
            new="new one\nnew two",
        )

        assert updated == "start\nnew one\nnew two\nend\n"

    def test_delete_region(self, scratchpad_dir: Path) -> None:
        """Delete content by supplying an empty new value."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("keep\nremove me\nkeep too\n", encoding="utf-8")

        updated = edit_file(path, pre="keep", post="keep too", original="remove me", new="")

        assert updated == "keep\nkeep too\n"

    def test_insert_into_empty_file(self, scratchpad_dir: Path) -> None:
        """Insert into an empty file using BOF and EOF anchors."""
        path = scratchpad_dir / PROGRESS_FILENAME
        path.write_text("", encoding="utf-8")

        updated = edit_file(path, pre=BOF_MARKER, post=EOF_MARKER, original="", new="first note")

        assert updated == "first note"
        assert read_file(path) == "first note"

    def test_insert_at_file_start(self, scratchpad_dir: Path) -> None:
        """Insert before the first line."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("existing\n", encoding="utf-8")

        updated = edit_file(path, pre=BOF_MARKER, post="existing", original="", new="intro")

        assert updated == "intro\nexisting\n"

    def test_insert_at_file_end(self, scratchpad_dir: Path) -> None:
        """Append after the last line."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("existing\n", encoding="utf-8")

        updated = edit_file(path, pre="existing", post=EOF_MARKER, original="", new="tail")

        assert updated == "existing\ntail\n"

    def test_apply_edit_raises_when_no_match(self) -> None:
        """Reject edits that do not match anchors."""
        with pytest.raises(AnchoredEditError, match="no region matched"):
            apply_edit("alpha\nbeta\n", pre="alpha", post="missing", original="beta", new="x")

    def test_apply_edit_raises_when_ambiguous(self) -> None:
        """Reject edits that match more than one region."""
        with pytest.raises(AnchoredEditError, match="multiple regions matched"):
            apply_edit(
                "b\nx\nb\nx\nb",
                pre="b",
                post="b",
                original="x",
                new="y",
            )


class TestAnchoredEditTools:
    """Agent tool wrapper tests."""

    def test_edit_scratchpad_tool_applies_delta(
        self,
        scratchpad_settings: Settings,
        scratchpad_dir: Path,
    ) -> None:
        """Apply anchored edits to SCRATCHPAD.md through the LangChain tool."""
        path = scratchpad_dir / SCRATCHPAD_FILENAME
        path.write_text("todo: check inbox\n", encoding="utf-8")
        tool = create_scratchpad_tools(scratchpad_settings)[0]

        result = tool.invoke(
            {
                "pre": BOF_MARKER,
                "post": EOF_MARKER,
                "original": "todo: check inbox",
                "new": "done: check inbox",
            },
        )

        assert result == "Scratchpad updated."
        assert read_file(path) == "done: check inbox\n"

    def test_edit_progress_tool_applies_delta(
        self,
        scratchpad_settings: Settings,
        scratchpad_dir: Path,
    ) -> None:
        """Apply anchored edits to PROGRESS.md through the LangChain tool."""
        path = scratchpad_dir / PROGRESS_FILENAME
        path.write_text("status: started\n", encoding="utf-8")
        tool = create_progress_tools(scratchpad_settings)[0]

        result = tool.invoke(
            {
                "pre": BOF_MARKER,
                "post": EOF_MARKER,
                "original": "status: started",
                "new": "status: in progress",
            },
        )

        assert result == "Progress journal updated."
        assert read_file(path) == "status: in progress\n"
