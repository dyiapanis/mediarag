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
        return subprocess.CompletedProcess(cmd, 0, stdout=b"200\t\ttext/plain; charset=utf-8", stderr=b"")

    real_run, real_which = tools.subprocess.run, shutil.which
    tools.subprocess.run = fake_run
    shutil.which = lambda name: "/usr/bin/curl"
    try:
        res = tools.fetch_url("https://example.com/readme")
        check("fetch ok", res["code"] == "ok", res)
        cmd = seen["cmd"]
        check("--resolve passed", "--resolve" in cmd, cmd)
        check("pinned IP reaches curl",
              cmd[cmd.index("--resolve") + 1] == "example.com:443:93.184.216.34", cmd)
        check("content type names an extensionless download", res["path"].endswith(".txt"), res["path"])
        check("content type file exists", os.path.isfile(res["path"]))
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

# --- malformed port is a structured refusal, not an uncaught ValueError ------
urlsafe._host_is_safe_url = lambda: None
pin, why = urlsafe.resolve_for_fetch("http://example.com:bad/")
check("malformed port refused cleanly", pin is None and why == "unparseable port in URL", why)

# --- every audio type we map must be one the indexer actually accepts --------
from core import ingest as _ing  # noqa: E402

for ctype, ext in tools._CONTENT_EXT.items():
    if ctype.startswith("audio/") or ctype == "application/ogg":
        check(f"{ctype} maps to an ingestible extension",
              _ing.kind_of("download" + ext) is not None, (ctype, ext))

# --- REAL curl: a dotted hostname must not escape the pin --------------------
# The pin used to be keyed on the dot-stripped host while curl still received the
# dotted URL, so curl ignored the pin and re-resolved. These two tests use a real
# loopback HTTP server: the pin points at 127.0.0.2 (nothing listens there) while
# the real server sits on 127.0.0.1. Any successful fetch would mean the pin lost.
if shutil.which("curl"):
    import http.server
    import threading

    hits = {"secret": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", f"http://localhost.:{port}/secret.txt")
                self.end_headers()
            elif self.path == "/secret.txt":
                hits["secret"] += 1
                body = b"secret"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def split_dns(host, port, *rest):
        ip = "127.0.0.1" if host.startswith("example.com") else "127.0.0.2"
        return [(2, 1, 6, "", (ip, 0))]

    try:
        os.environ["MEDIARAG_ALLOW_PRIVATE_URLS"] = "1"  # loopback allowed, so the test is about pinning
        urlsafe.socket.getaddrinfo = split_dns
        res = tools.fetch_url(f"http://localhost.:{port}/secret.txt")
        check("dotted host cannot escape the pin", res["code"] != "ok", res)
        check("direct dotted fetch never reached the server", hits["secret"] == 0, hits)

        res = tools.fetch_url(f"http://example.com:{port}/start")
        check("dotted redirect target is refused too", res["code"] != "ok", res)
        check("redirect never reached the secret", hits["secret"] == 0, hits)
    finally:
        srv.shutdown()
        srv.server_close()
        urlsafe.socket.getaddrinfo = real_getaddrinfo
        os.environ.pop("MEDIARAG_ALLOW_PRIVATE_URLS", None)
else:
    print("SKIP real-curl dotted-host checks (curl not installed)")

print("\nALL URL PINNING CHECKS PASS")
