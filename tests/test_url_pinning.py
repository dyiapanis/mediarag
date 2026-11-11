"""URL fetch pinning: the connection must use the IP that passed validation.

Regression for the DNS-rebinding TOCTOU finding (t_a5e70bcd): validation used
Python's DNS, then curl re-resolved the hostname and could be handed a second,
hostile address.

Run: python tests/test_url_pinning.py   (offline: DNS and curl are stubbed)
"""
import os
import shutil
import subprocess
import sys
import types

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import tools, urlsafe  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


def stub_dns(ip):
    def fake(host, port, *rest):
        return [(2, 1, 6, "", (ip, 0))]
    return fake


real_getaddrinfo = urlsafe.socket.getaddrinfo
real_host_check = urlsafe._host_is_safe_url
urlsafe._host_is_safe_url = lambda: None  # standalone path for tests 1-4
try:
    # 1. the pin is the address that was validated, not whatever DNS says later
    urlsafe.socket.getaddrinfo = stub_dns("93.184.216.34")
    pin, why = urlsafe.resolve_for_fetch("https://example.com/doc")
    check("no problem for a public host", why is None, why)
    check("pin uses validated IP", pin == "example.com:443:93.184.216.34", pin)
    # A second resolution handing back a link-local address is refused outright.
    urlsafe.socket.getaddrinfo = stub_dns("169.254.169.254")
    check("rebind to link-local refused", urlsafe.url_problem("https://example.com/doc") is not None)

    # 2. private allowed still refuses metadata, even for a literal IP
    os.environ["MEDIARAG_ALLOW_PRIVATE_URLS"] = "1"
    check("metadata refused with private allowed",
          "link-local" in (urlsafe.url_problem("https://169.254.169.254/x") or ""))
    urlsafe.socket.getaddrinfo = stub_dns("192.168.1.10")
    check("private host allowed when opted in",
          urlsafe.url_problem("https://192.168.1.10/notes") is None)
    os.environ.pop("MEDIARAG_ALLOW_PRIVATE_URLS", None)

    # 3. IPv6 literals are bracketed for --resolve
    urlsafe.socket.getaddrinfo = stub_dns("2606:4700:4700::1111")
    pin, why = urlsafe.resolve_for_fetch("https://example.com/x")
    check("IPv6 bracketed", pin == "example.com:443:[2606:4700:4700::1111]", pin)
    pin, why = urlsafe.resolve_for_fetch("http://example.com:8080/x")
    check("explicit port preserved", pin == "example.com:8080:[2606:4700:4700::1111]", pin)

    # 4. fetch_url actually passes --resolve with the pinned address to curl
    urlsafe.socket.getaddrinfo = stub_dns("93.184.216.34")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = cmd[cmd.index("-o") + 1]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write("hello")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"200 ", stderr=b"")

    real_run, real_which = tools.subprocess.run, shutil.which
    tools.subprocess.run = fake_run
    shutil.which = lambda name: "/usr/bin/curl"
    try:
        res = tools.fetch_url("https://example.com/readme.md")
        check("fetch ok", res["code"] == "ok", res)
        cmd = seen["cmd"]
        check("--resolve passed", "--resolve" in cmd, cmd)
        check("pinned IP reaches curl",
              cmd[cmd.index("--resolve") + 1] == "example.com:443:93.184.216.34", cmd)
    finally:
        tools.subprocess.run, shutil.which = real_run, real_which

    # 5. Hermes-governed path: policy is not re-refused, but the IP is still pinned
    urlsafe._host_is_safe_url = lambda: (lambda url: True)
    urlsafe.socket.getaddrinfo = stub_dns("192.168.1.10")
    pin, why = urlsafe.resolve_for_fetch("https://intranet.local/notes")
    check("hermes-governed URL not re-refused", why is None, why)
    check("hermes path still pins", pin == "intranet.local:443:192.168.1.10", pin)
    urlsafe._host_is_safe_url = lambda: None
finally:
    urlsafe.socket.getaddrinfo = real_getaddrinfo
    urlsafe._host_is_safe_url = real_host_check
    os.environ.pop("MEDIARAG_ALLOW_PRIVATE_URLS", None)

print("\nALL URL PINNING CHECKS PASS")
