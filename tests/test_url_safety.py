"""Tests for URL safety validation."""

from __future__ import annotations

import pytest

from tools.webfetch.url_safety import UnsafeUrlError, validate_fetch_url


class TestValidateFetchUrl:
    """SSRF-safe URL validation tests."""

    def test_accepts_public_https_url(self) -> None:
        """Allow normal HTTPS URLs."""
        url = "https://example.com/article"
        assert validate_fetch_url(url) == url

    def test_rejects_non_http_scheme(self) -> None:
        """Block file and other schemes."""
        with pytest.raises(UnsafeUrlError, match="Unsupported URL scheme"):
            validate_fetch_url("file:///etc/passwd")

    def test_rejects_localhost(self) -> None:
        """Block localhost hostnames."""
        with pytest.raises(UnsafeUrlError, match="Localhost"):
            validate_fetch_url("http://localhost:8080/admin")

    def test_rejects_private_ip_literal(self) -> None:
        """Block private IP literals."""
        with pytest.raises(UnsafeUrlError, match="Private or local"):
            validate_fetch_url("http://192.168.1.10/page")

    def test_rejects_loopback_ip_literal(self) -> None:
        """Block loopback IP literals."""
        with pytest.raises(UnsafeUrlError, match="Private or local"):
            validate_fetch_url("http://127.0.0.1/secret")
