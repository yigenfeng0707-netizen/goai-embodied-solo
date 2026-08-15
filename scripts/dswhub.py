"""Authenticated client for the Aliyun DSW JupyterLab gateway (instance dsw-2098907).

Auth: single browser session cookie `login_aliyunid_ticket` extracted via CDP
(see aliyun-dsw skill cdp_ms.py). The cookie works for any instance on the
same account.
"""
from __future__ import annotations

import json
import os
import time
import requests
import websocket

DOMAIN = "https://dsw-gateway-cn-hangzhou.data.aliyun.com"
INSTANCE = "dsw-2103539"
BASE = DOMAIN + "/" + INSTANCE
COOKIE_CACHE = r"C:\Users\52637\.config\opencode\skills\aliyun-dsw-remote-work\ms_session_cookies.json"


def load_ticket(cache_path: str = COOKIE_CACHE) -> str:
    data = json.load(open(os.path.abspath(cache_path), encoding="utf-8"))
    return data["cookies"]["login_aliyunid_ticket"]


class Dswhub:
    def __init__(self, ticket: str | None = None, cache_path: str = COOKIE_CACHE):
        self.ticket = ticket or load_ticket(cache_path)
        self.s = requests.Session()
        self.s.headers.update({"Cookie": f"login_aliyunid_ticket={self.ticket}"})

    def _get(self, path, **kw):
        r = self.s.get(BASE + path, timeout=30, allow_redirects=False, **kw)
        if r.status_code in (301, 302) and "login" in r.headers.get("Location", ""):
            raise RuntimeError("session expired - re-extract cookie via cdp_ms.py")
        r.raise_for_status()
        return r

    def me(self):
        return self._get("/api/me").json()

    def contents(self, path: str = "", content: int = 1):
        return self._get(f"/api/contents/{path}", params={"content": content}).json()

    def kernels(self):
        return self._get("/api/kernels").json()

    def _xsrf(self, cache_path: str = COOKIE_CACHE) -> str:
        data = json.load(open(os.path.abspath(cache_path), encoding="utf-8"))
        c = data.get("cookies", {})
        return c.get("_xsrf") or c.get("XSRF-TOKEN") or ""

    def run_python(self, code: str, timeout: int = 300) -> str:
        """Headless kernel execution via REST+WebSocket (no browser needed)."""
        xsrf = self._xsrf()
        # create a kernel
        r = self.s.post(BASE + "/api/kernels",
                        headers={"X-XSRFToken": xsrf}, timeout=30)
        r.raise_for_status()
        kid = r.json()["id"]
        buf = ""
        try:
            ws_url = "wss://" + "dsw-gateway-cn-hangzhou.data.aliyun.com" + "/" + INSTANCE + "/api/kernels/" + kid + "/channels"
            ws = websocket.create_connection(
                ws_url, timeout=60,
                header=[
                    "Cookie: login_aliyunid_ticket=" + self.ticket + "; _xsrf=" + xsrf,
                    "X-XSRFToken: " + xsrf,
                    "Origin: " + DOMAIN,
                ],
            )
            msg_id = "exec_" + str(int(time.time() * 1000))
            ws.send(json.dumps({
                "header": {"msg_id": msg_id, "username": "user", "session": "s1",
                           "msg_type": "execute_request", "version": "5.3"},
                "parent_header": {}, "metadata": {},
                "content": {"code": code, "silent": False, "store_history": False,
                            "user_expressions": {}, "allow_stdin": False,
                            "stop_on_error": False},
                "channel": "shell",
            }))
            ws.settimeout(timeout + 30)
            t0 = time.time()
            while True:
                raw = ws.recv()
                msg = json.loads(raw)
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                t = msg.get("msg_type")
                if t == "stream":
                    buf += msg["content"]["text"]
                elif t == "execute_result":
                    buf += msg["content"]["data"].get("text/plain", "")
                elif t == "display_data":
                    buf += msg["content"]["data"].get("text/plain", "")
                elif t == "error":
                    buf += "PYERR: " + "\n".join(msg["content"]["traceback"])
                elif t == "status" and msg["content"]["execution_state"] == "idle":
                    break
                if time.time() - t0 > timeout + 30:
                    buf += "\n[TIMEOUT]"
                    break
            try:
                ws.close()
            except Exception:
                pass
            return buf
        finally:
            try:
                self.s.delete(BASE + "/api/kernels/" + kid,
                               headers={"X-XSRFToken": xsrf}, timeout=20)
            except Exception:
                pass


if __name__ == "__main__":
    d = Dswhub()
    print("me:", d.me().get("identity"))
    print("contents root:", [c["name"] for c in d.contents("")["content"]][:12])
