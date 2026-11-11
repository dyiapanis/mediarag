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


# The single URLSAFE_HOST_CHARS set guards a manually-extracted raw host. .hostname is unusable for
# this: it lowercases (turning U+212A KELVIN SIGN into ASCII 'k') and would let
# non-ASCII sneak past the guard only to be canonicalised differently by curl.
URLSAFE_HOST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-:")


def _raw_host(parts) -> str:
    """The authority as written, before ANY normalisation."""
    netloc = parts.netloc or ""
    if "@" in netloc:  # userinfo rejected separately; never its business here
        netloc = netloc.rsplit("@", 1)[1]
    if netloc.startswith("[") and "]" in netloc:  # IPv6 literal without its brackets
        return netloc[1:netloc.index("]")]
    return netloc.rsplit(":", 1)[0] if ":" in netloc else netloc


def _resolve(url: str) -> tuple[list[str], Optional[str]]:
    """Validate the URL and return the IPs we are willing to connect to.

    Resolution happens here exactly once, so the address that gets validated is
    the address the fetch can later be pinned to. Returning the IPs alongside
    the verdict is what closes the DNS-rebinding TOCTOU hole.
    """
    url = url or ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return [], "unparseable URL"
    raw_host = _raw_host(parts)
    # Guard the authority AS WRITTEN, before any normalisation: Python lowercases
    # U+212A (KELVIN SIGN) to ASCII 'k' via urlsplit, so a normalised host could
    # pass here and then be canonicalised differently by curl — which is how a
    # pin stops pinning.
    if any(ch not in URLSAFE_HOST_CHARS for ch in raw_host):
        # curl canonicalises non-ASCII hosts to punycode, so a Unicode --resolve
        # key can never match. Until we canonicalise the exact way curl does,
        # such URLs must be refused — a pin that cannot match is no pin at all.
        return [], "non-ASCII or invalid hostname characters are not supported"
    host = raw_host.rstrip(".").lower()
    if parts.scheme.lower() not in ("http", "https"):
        return [], f"only http(s) URLs are allowed, got {parts.scheme or 'no scheme'!r}"
    if not host or "@" in parts.netloc:
        return [], "URL must have a plain host (no userinfo)"
    if host in _METADATA_HOSTS:
        return [], "cloud-metadata host"
    allow_private = os.environ.get("MEDIARAG_ALLOW_PRIVATE_URLS") == "1"
    hermes_check = None if allow_private else _host_is_safe_url()
    if hermes_check is not None and not hermes_check(url):
        return [], "blocked by Hermes URL safety policy"
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return [], f"could not resolve host {host!r}"
    ips = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            return [], "unparseable resolved address"
        if hermes_check is None:
            why = _ip_problem(ip, allow_private)
            if why:
                return [], f"{host}: {why}"
        # When Hermes governs the policy we do not re-refuse on its behalf, but
        # we still record the resolved address so the fetch can be pinned to it.
        ips.append(str(ip))
    return ips, None


def url_problem(url: str) -> Optional[str]:
    """None if `url` may be fetched, otherwise a short reason."""
    return _resolve(url)[1]


def resolve_for_fetch(url: str) -> tuple[Optional[str], Optional[str]]:
    """A curl --resolve argument pinning the connection to the validated IP.

    Without this, curl re-resolves the hostname itself and can be handed a
    different address than the one that just passed validation (rebinding).
    """
    ips, problem = _resolve(url)
    if problem:
        return None, problem
    if not ips:
        return None, "no usable resolved address"
    parts = urlsplit(url)
    raw_host = _raw_host(parts)
    if any(ch not in URLSAFE_HOST_CHARS for ch in raw_host):
        return None, "non-ASCII or invalid hostname characters are not supported"
    plain = raw_host.lower()  # dot preserved EXACTLY as written: round-1's fix
    host_key = f"[{plain}]" if ":" in plain else plain  # IPv6 keys need brackets or curl ignores the pin
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError:
        return None, "unparseable port in URL"
    ip = ips[0]
    literal = f"[{ip}]" if ":" in ip else ip  # IPv6 needs brackets in --resolve
    return f"{host_key}:{port}:{literal}", None
