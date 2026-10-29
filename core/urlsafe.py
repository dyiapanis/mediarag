"""URL policy for add_source http(s) fetches: public hosts only, checked on every redirect hop.

Inside Hermes the host's own SSRF check (tools.url_safety.is_safe_url) decides, so the
user's Hermes `security.allow_private_urls` setting applies. Standalone, a local check
with the same rules is used. Loopback/private hosts are allowed only with
MEDIARAG_ALLOW_PRIVATE_URLS=1; link-local and cloud-metadata hosts are always refused.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from typing import Optional
from urllib.parse import urlsplit

_METADATA_HOSTS = {"metadata.google.internal", "metadata.goog"}
_METADATA_IPS = {ipaddress.ip_address(a) for a in (
    "169.254.169.254", "169.254.170.2", "169.254.169.253", "100.100.100.200", "fd00:ec2::254")}
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _host_is_safe_url():
    try:
        from tools.url_safety import is_safe_url  # present when running inside Hermes
        return is_safe_url
    except Exception:
        return None


def _ip_problem(ip, allow_private: bool) -> Optional[str]:
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    if ip in _METADATA_IPS or ip.is_link_local:
        return "link-local / cloud-metadata address"
    if allow_private:
        return None
    if (ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_multicast
            or ip.is_unspecified or ip in _CGNAT):
        return "loopback/private address (set MEDIARAG_ALLOW_PRIVATE_URLS=1 to allow)"
    return None


def url_problem(url: str) -> Optional[str]:
    """None if `url` may be fetched, otherwise a short reason."""
    try:
        parts = urlsplit((url or "").strip())
        host = (parts.hostname or "").rstrip(".").lower()
    except ValueError:
        return "unparseable URL"
    if parts.scheme.lower() not in ("http", "https"):
        return f"only http(s) URLs are allowed, got {parts.scheme or 'no scheme'!r}"
    if not host or "@" in parts.netloc:
        return "URL must have a plain host (no userinfo)"
    if host in _METADATA_HOSTS:
        return "cloud-metadata host"
    allow_private = os.environ.get("MEDIARAG_ALLOW_PRIVATE_URLS") == "1"
    if not allow_private:
        hermes_check = _host_is_safe_url()
        if hermes_check is not None:
            return None if hermes_check(url) else "blocked by Hermes URL safety policy"
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return f"could not resolve host {host!r}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            return "unparseable resolved address"
        why = _ip_problem(ip, allow_private)
        if why:
            return f"{host}: {why}"
    return None
