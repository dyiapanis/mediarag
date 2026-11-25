"""Dual-era stdio protocol conformance (MCP 2026-07-28 modern + legacy handshake).

Direct-run: python3 tests/test_stdio_protocol.py (exit 0 = all pass).
Spawns scripts/mcp_server.py and drives both eras over real stdio framing:
modern per-request _meta versioning (server/discover, resultType, -32022) and
the legacy initialize handshake (version echo, unchanged response shapes).
"""
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODERN = "2026-07-28"
_VER_KEY = "io.modelcontextprotocol/protocolVersion"


class Server:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(_ROOT, "scripts", "mcp_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=_ROOT)
        assert self.proc.stdin is not None and self.proc.stdout is not None

    def rpc(self, method, rid=1, params=None):
        line = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                           **({"params": params} if params is not None else {})})
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def main():
    checks = []

    def check(label, ok):
        checks.append(ok)
        print(("PASS" if ok else "FAIL"), label)

    s = Server()
    try:
        # -- modern era: per-request _meta versioning --------------------------
        d = s.rpc("server/discover", rid=10, params={
            "_meta": {_VER_KEY: _MODERN,
                      "io.modelcontextprotocol/clientInfo": {"name": "probe", "version": "0"}}})
        r = d.get("result", {})
        check("discover: supportedVersions+capabilities+serverInfo",
              r.get("resultType") == "complete"
              and r.get("supportedVersions") == [_MODERN]
              and "tools" in r.get("capabilities", {})
              and r.get("_meta", {}).get("io.modelcontextprotocol/serverInfo", {}).get("name") == "mediarag")

        tl = s.rpc("tools/list", rid=11, params={"_meta": {_VER_KEY: _MODERN}})
        r = tl.get("result", {})
        check("modern tools/list: resultType+ttlMs+5 tools",
              r.get("resultType") == "complete" and "ttlMs" in r
              and len(r.get("tools", [])) == 5)

        ping = s.rpc("ping", rid=12, params={"_meta": {_VER_KEY: _MODERN}})
        check("modern ping ok", ping.get("result") == {})

        bad = s.rpc("tools/list", rid=13, params={"_meta": {_VER_KEY: "1900-01-01"}})
        e = bad.get("error", {})
        check("-32022 with requested+supported",
              e.get("code") == -32022
              and e.get("data", {}).get("requested") == "1900-01-01"
              and e.get("data", {}).get("supported") == [_MODERN])

        # -- legacy era: initialize handshake, response shapes unchanged -------
        init = s.rpc("initialize", rid=20, params={"protocolVersion": "2024-11-05",
                                                   "capabilities": {}})
        r = init.get("result", {})
        check("legacy initialize echoes requested 2024-11-05",
              r.get("protocolVersion") == "2024-11-05" and "serverInfo" in r)

        init2 = s.rpc("initialize", rid=21, params={"protocolVersion": "2025-11-25",
                                                    "capabilities": {}})
        check("legacy initialize echoes 2025-11-25",
              init2.get("result", {}).get("protocolVersion") == "2025-11-25")

        tl2 = s.rpc("tools/list", rid=22)
        r = tl2.get("result", {})
        check("legacy tools/list: 5 tools, NO modern fields",
              len(r.get("tools", [])) == 5
              and "resultType" not in r and "cacheScope" not in r)

        unk = s.rpc("tools/call", rid=23, params={"name": "nope", "arguments": {}})
        check("legacy unknown tool -32601",
              unk.get("error", {}).get("code") == -32601)

        # -- unversioned server/discover (the stdio fallback probe) ------------
        d2 = s.rpc("server/discover", rid=30)
        check("unversioned discover still answered",
              d2.get("result", {}).get("supportedVersions") == [_MODERN])

        mping = s.rpc("ping", rid=31, params={"_meta": {}})
        check("_meta without version key = legacy ping", mping.get("result") == {})
    finally:
        s.close()

    print(f"\n{sum(checks)}/{len(checks)} pass")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
