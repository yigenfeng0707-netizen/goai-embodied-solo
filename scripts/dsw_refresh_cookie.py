"""One-click refresh of the DSW gateway session cookie.

Why this is needed: dswhub.py authenticates via a single browser cookie
`login_aliyunid_ticket` stored in ms_session_cookies.json. That cookie expires
after a few hours, after which every dswhub call 404s and monitoring goes blind.

The naive CDP extraction (cdp_ms.py) fails on this machine because:
  1. Launching Chrome with the *real* Default profile + --remote-debugging-port
     does not open the port (singleton lock / profile conflict).
  2. The login cookie is a PARTITIONED cookie (CHIPS); Storage.getCookies
     without a partitionKey does not return it.

This script works around both:
  - kill Chrome, copy the Default profile to a temp dir (no singleton lock),
    launch the COPY with the debug port  ->  port opens reliably.
  - extract cookies for the gateway domain AND with partitionKey topLevelSite,
    so login_aliyunid_ticket is captured.
  - restore the user's real Chrome afterwards (--restore-last-session) so their
    session is not disrupted.
  - verify by calling dswhub.me().

The local keepalive / ckpt-watcher re-read ms_session_cookies.json on every
call, so refreshing this file alone makes them resume automatically.

Usage:
    python dsw_refresh_cookie.py
"""
from __future__ import annotations
import os, sys, time, json, subprocess, shutil
import requests, websocket

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dswhub  # provides INSTANCE, DOMAIN, COOKIE_CACHE

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME):
    CHROME = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
USER_DATA = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")
SRC_PROFILE = os.path.join(USER_DATA, "Default")
DST_PROFILE = r"C:\Users\52637\AppData\Local\Temp\chrome_profile_copy"
PORT = 9222
COOKIE_CACHE = dswhub.COOKIE_CACHE
DSW_URL = f"{dswhub.DOMAIN}/dsw-{dswhub.INSTANCE}/lab"
GW = dswhub.DOMAIN  # https://dsw-gateway-cn-hangzhou.data.aliyun.com
LOG = os.path.join(HERE, "dsw_refresh_cookie.log")

def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")
    print(m, flush=True)

def port_up():
    try:
        return requests.get(f"http://127.0.0.1:{PORT}/json/version", timeout=2).status_code == 200
    except Exception:
        return False

def kill_debug_chrome():
    # Kill only Chrome instances we launched with the debug port (not the user's real one).
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
         "Where-Object { $_.CommandLine -like '*remote-debugging-port*' } | "
         "ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True).stdout.split()
    for pid in out:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, text=True)
    time.sleep(2)

def launch_copy_chrome():
    log("copying Default profile -> %s" % DST_PROFILE)
    if os.path.exists(DST_PROFILE):
        shutil.rmtree(DST_PROFILE, ignore_errors=True)
    # Copy the profile but EXCLUDE heavy cache dirs (otherwise robocopy crawls
    # 100k+ files and can appear to hang). Cookies live in small SQLite files
    # under Default/, which are copied fine without the caches.
    EXCL = ["Cache", "Code Cache", "GPUCache", "Service Worker", "Session Storage",
            "IndexedDB", "blob_storage", "optimization_guide_model_store", "Crashpad",
            "GrShaderCache", "Subresource Filter", "OnDeviceHeadSuggestModel"]
    args = ["robocopy", SRC_PROFILE, DST_PROFILE, "/E", "/R:1", "/W:1", "/XJ", "/NFL", "/NDL", "/NJH"]
    for e in EXCL:
        args += ["/XD", e]
    rc = subprocess.run(args, capture_output=True, text=True).returncode
    for lf in [os.path.join(DST_PROFILE, "Lock"), os.path.join(DST_PROFILE, "SingletonLock")]:
        try: os.remove(lf)
        except Exception: pass
    log("robocopy rc=%s" % rc)
    subprocess.Popen([CHROME, f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
                      f"--user-data-dir={DST_PROFILE}", "--no-first-run", "--no-default-browser-check",
                      "--restore-last-session", DSW_URL])
    for i in range(40):
        time.sleep(1)
        if port_up():
            log("debug port up after %ds" % (i + 1)); return True
    log("debug port NOT up"); return False

def get_ws():
    for _ in range(15):
        try:
            ts = requests.get(f"http://127.0.0.1:{PORT}/json", timeout=3).json()
            for t in ts:
                if t.get("type") == "page":
                    return t["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(1)
    raise RuntimeError("no page target on debug port")

class CDP:
    def __init__(self, ws):
        self.ws = websocket.create_connection(ws, timeout=30); self._id = 0
    def call(self, m, p=None):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": m, "params": p or {}}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self._id:
                return r.get("result", {})
    def cookies(self, urls=None, pk=None):
        params = {}
        if urls is not None: params["urls"] = urls
        if pk is not None: params["partitionKey"] = pk
        res = self.call("Storage.getCookies", params)
        return {c["name"]: c["value"] for c in res.get("cookies", [])}

def restore_user_chrome():
    # Reopen the user's real Chrome so their session is uninterrupted.
    subprocess.Popen([CHROME, f"--user-data-dir={USER_DATA}", "--restore-last-session",
                      "--no-first-run", "--no-default-browser-check"])
    log("relaunched user's real Chrome")

def main():
    open(LOG, "w", encoding="utf-8").close()
    log("=== dsw_refresh_cookie for instance %s ===" % dswhub.INSTANCE)
    kill_debug_chrome()
    if not launch_copy_chrome():
        log("FAILED: could not open debug port"); return
    try:
        ws = get_ws(); cdp = CDP(ws); log("cdp connected")
        cdp.call("Page.enable"); cdp.call("Page.navigate", {"url": DSW_URL}); log("navigated"); time.sleep(8)
        merged = {}
        try:
            merged.update(json.load(open(COOKIE_CACHE, encoding="utf-8")).get("cookies", {}))
        except Exception:
            pass
        merged.update(cdp.cookies(urls=[DSW_URL]))
        merged.update(cdp.cookies(urls=["https://account.aliyun.com"]))
        merged.update(cdp.cookies(urls=["https://www.modelscope.cn", "https://modelscope.cn"]))
        merged.update(cdp.cookies(urls=[DSW_URL], pk={"topLevelSite": GW}))
        merged.update(cdp.cookies(pk={"topLevelSite": GW}))
        merged.update(cdp.cookies(pk={"topLevelSite": "https://account.aliyun.com"}))
        cdp.ws.close()
        json.dump({"domain": DSW_URL, "cookies": merged,
                   "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(COOKIE_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log("SAVED %d cookies; has login_aliyunid_ticket: %s" %
            (len(merged), "login_aliyunid_ticket" in merged))
    finally:
        kill_debug_chrome()  # close the temp copy-profile chrome
        restore_user_chrome()
    # verify
    try:
        ident = dswhub.Dswhub().me().get("identity")
        log("VERIFY me() OK -> %s" % ident)
    except Exception as e:
        log("VERIFY FAILED: %r" % e)

if __name__ == "__main__":
    main()
