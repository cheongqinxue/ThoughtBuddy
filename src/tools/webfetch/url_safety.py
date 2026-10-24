"""URL validation helpers for outbound HTTP fetches."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_HOSTNAMES = frozenset({"localhost"})
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeUrlError(ValueError):
    """Raised when a URL must not be fetched."""


def validate_fetch_url(url: str) -> str:
    """Validate that a URL is safe to fetch over HTTP(S).

    Blocks non-HTTP schemes, missing hosts, localhost, and private or
    link-local IP addresses (SSRF mitigation).

    Args:
        url: Candidate URL.

    Returns:
        The original URL when validation succeeds.

    Raises:
        UnsafeUrlError: If the URL is not allowed.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in _ALLOWED_SCHEMES:
        msg = f"Unsupported URL scheme: {parsed.scheme or 'missing'}"
        raise UnsafeUrlError(msg)

    hostname = parsed.hostname
    if not hostname:
        raise UnsafeUrlError("URL must include a hostname.")

    normalized_host = hostname.removeprefix("[").removesuffix("]")
    if normalized_host.lower() in _BLOCKED_HOSTNAMES:
        raise UnsafeUrlError("Localhost URLs are not allowed.")

    try:
        literal = ipaddress.ip_address(normalized_host)
    except ValueError:
        _validate_resolved_addresses(normalized_host)
    else:
        _reject_unsafe_ip(literal)

    return url


def _validate_resolved_addresses(hostname: str) -> None:
    """Resolve a hostname and reject unsafe target addresses.

    Args:
        hostname: Hostname to resolve.

    Raises:
        UnsafeUrlError: If resolution fails or yields unsafe addresses.
    """
    try:
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        msg = f"Could not resolve hostname: {hostname}"
        raise UnsafeUrlError(msg) from exc

    if not addr_infos:
        raise UnsafeUrlError(f"Could not resolve hostname: {hostname}")

    for _, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        _reject_unsafe_ip(ip)


def _reject_unsafe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Reject private, loopback, link-local, or reserved IP targets.

    Args:
        ip: Resolved IP address.

    Raises:
        UnsafeUrlError: If the address is not safe to fetch.
    """
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    ):
        raise UnsafeUrlError("Private or local network URLs are not allowed.")
