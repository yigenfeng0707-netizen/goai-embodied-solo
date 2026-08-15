"""Connect to DSW, report UnifiedACT/cloudflared status, recover if needed."""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, r"c:\Users\52637\.trae-cn\skills\dsw-remote-execution\scripts")
sys.path.insert(0, r"d:\APPs\GOAI_OpenSource\goai-embodied-solo\scripts")

from dsw_remote import DswRemote, ensure_chrome_with_cdp, CDP_PORT
from common_config import DSW_URL


def _is_login_url(url: str) -> bool:
    u = url.lower()
    return any(k in u for k in ("login", "account.aliyun", "signin", "passport.aliyun"))


def _is_lab_url(url: str) -> bool:
    u = url.lower()
    return "dsw-gateway" in u and "/lab" in u and not _is_login_url(u) and "chrome-error" not in u


async def wait_for_lab(dsw: DswRemote, timeout_sec: int = 300) -> None:
    """Ensure page is a real JupyterLab URL (not login / chrome-error)."""
    # Prefer an already-open lab tab
    for pg in dsw.context.pages:
        if _is_lab_url(pg.url):
            dsw.page = pg
            print(f"[DSW] reusing lab tab: {pg.url[:100]}")
            return

    if dsw.page is None:
        dsw.page = await dsw.context.new_page()

    print(f"[DSW] navigating to {DSW_URL}")
    try:
        await dsw.page.goto(DSW_URL, wait_until="domcontentloaded", timeout=90000)
    except Exception as e:
        print(f"[DSW] goto warning: {e}")

    for i in range(timeout_sec):
        cur = dsw.page.url
        if _is_lab_url(cur):
            # Confirm Jupyter API responds
            base = cur.split("/lab")[0]
            try:
                ok = await dsw.page.evaluate(
                    """async (base) => {
                        try {
                          const r = await fetch(`${base}/api/status`);
                          return r.ok;
                        } catch (e) { return false; }
                    }""",
                    base,
                )
            except Exception:
                ok = False
            if ok:
                print(f"[DSW] lab ready: {cur[:100]}")
                return
            print(f"[DSW] lab URL but API not ready yet... ({i+1}s)")
        elif _is_login_url(cur):
            if i == 0 or i % 30 == 29:
                print("[DSW] login required — please sign in on the Chrome window")
                print(f"  waiting... ({i+1}s) url={cur[:80]}")
        else:
            if i % 15 == 0:
                print(f"[DSW] waiting for lab... ({i+1}s) url={cur[:100]}")
                # Retry navigation periodically if stuck on chrome-error
                if "chrome-error" in cur.lower() or i % 60 == 59:
                    try:
                        await dsw.page.goto(DSW_URL, wait_until="domcontentloaded", timeout=60000)
                    except Exception as e:
                        print(f"[DSW] renav warning: {e}")
        await asyncio.sleep(1)

    raise RuntimeError(f"lab not ready after {timeout_sec}s; last url={dsw.page.url}")


async def run(dsw, cmd, timeout=60, label=""):
    if label:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    out = await dsw.run_shell(cmd, timeout=timeout)
    print(out, flush=True)
    return out or ""


async def main():
    print(f"[1] Connecting to DSW: {DSW_URL}", flush=True)
    if not ensure_chrome_with_cdp():
        raise RuntimeError("Chrome CDP failed")

    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{CDP_PORT}")
    context = browser.contexts[0]

    dsw = DswRemote(dsw_url=DSW_URL)
    dsw._playwright = pw
    dsw.browser = browser
    dsw.context = context
    dsw.page = None

    await wait_for_lab(dsw)
    print("Connected.", flush=True)

    try:
        await run(
            dsw,
            "hostname; date; echo '---'; "
            "df -h /mnt/workspace 2>&1 | tail -2; "
            "echo '---'; "
            "ls -la /mnt/workspace/goai_recover.sh 2>&1; "
            "test -d /mnt/workspace/RoboDojo && echo ROBODOJO_OK || echo ROBODOJO_MISSING",
            timeout=45,
            label="A. Instance + workspace",
        )

        status = await run(
            dsw,
            "echo '=== PROCESSES ==='; "
            "ps aux | grep -E 'setup_policy_server|cloudflared' | grep -v grep || echo 'NO_POLICY_OR_TUNNEL'; "
            "echo '=== PORTS ==='; "
            "ss -tlnp 2>&1 | grep -E '1900[0-9]|8443' || echo 'NO_POLICY_PORTS'; "
            "echo '=== GPU ==='; "
            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1 || echo 'NO_GPU'",
            timeout=45,
            label="B. Processes / ports / GPU",
        )

        need_recover = (
            "NO_POLICY_OR_TUNNEL" in status
            or "NO_POLICY_PORTS" in status
            or "setup_policy_server" not in status
        )

        await run(
            dsw,
            "echo '=== cotrain unified ckpt ==='; "
            "d=/mnt/workspace/RoboDojo/ckpt/ACT/act-RoboDojo-cotrain/arx_x5-100-joint; "
            "if [ -d \"$d\" ]; then "
            "  n=$(ls \"$d\"/policy_*.ckpt 2>/dev/null | wc -l); "
            "  echo \"cotrain: OK ($n ckpt files) at $d\"; "
            "else echo \"cotrain: MISSING ($d)\"; fi; "
            "echo '=== UnifiedACT ==='; "
            "ls /mnt/workspace/RoboDojo/XPolicyLab/policy/UnifiedACT/ 2>&1 | head -20",
            timeout=45,
            label="C. Checkpoints + UnifiedACT",
        )

        await run(
            dsw,
            "echo '--- cloudflared.log ---'; "
            "tail -30 /mnt/workspace/cloudflared.log 2>&1 || echo 'no cloudflared.log'; "
            "echo '--- unifiedact_server.log ---'; "
            "tail -20 /mnt/workspace/unifiedact_server.log 2>&1 || echo 'no unifiedact_server.log'",
            timeout=30,
            label="D. Logs / tunnel URL",
        )

        if need_recover:
            print("\n[RECOVER] UnifiedACT/tunnel not healthy — running goai_recover.sh", flush=True)
            exists = await run(
                dsw,
                "test -f /mnt/workspace/goai_recover.sh && echo RECOVER_EXISTS || echo RECOVER_MISSING",
                timeout=20,
                label="E0. Recover script present?",
            )
            if "RECOVER_EXISTS" in exists:
                await run(
                    dsw,
                    "bash /mnt/workspace/goai_recover.sh 2>&1",
                    timeout=180,
                    label="E. Recovery",
                )
            else:
                print("[RECOVER] goai_recover.sh missing — cannot auto-recover", flush=True)

            await run(
                dsw,
                "echo '=== AFTER RECOVER ==='; "
                "ps aux | grep -E 'setup_policy_server|cloudflared' | grep -v grep || echo 'STILL_DOWN'; "
                "ss -tlnp 2>&1 | grep 19002 || echo 'PORT_19002_DOWN'; "
                "grep -oE 'https://[a-zA-Z0-9.-]+\\.trycloudflare\\.com' "
                "/mnt/workspace/cloudflared.log 2>&1 | tail -3",
                timeout=40,
                label="F. Post-recover verify",
            )
        else:
            print("\n[OK] UnifiedACT/tunnel appear running — skipping recover", flush=True)
            await run(
                dsw,
                "grep -oE 'https://[a-zA-Z0-9.-]+\\.trycloudflare\\.com' "
                "/mnt/workspace/cloudflared.log 2>&1 | tail -3; "
                "PYTHONPATH=/mnt/workspace/RoboDojo/XPolicyLab timeout 20 python3 -c \""
                "import sys; sys.path.insert(0,'/mnt/workspace/RoboDojo/XPolicyLab'); "
                "from client_server.ws.model_client import WsModelClient; "
                "import numpy as np; "
                "c=WsModelClient(url='ws://localhost:19002',evaluation_id='h',trial_id='h',action_case_id='h'); "
                "c.call(func_name='reset'); "
                "obs={'vision':{'cam_head':{'color':np.zeros((480,640,3),dtype=np.uint8)},"
                "'cam_left_wrist':{'color':np.zeros((480,640,3),dtype=np.uint8)},"
                "'cam_right_wrist':{'color':np.zeros((480,640,3),dtype=np.uint8)}},"
                "'state':{'left_arm_joint_state':np.zeros(6,dtype=np.float32),"
                "'left_ee_joint_state':np.zeros(1,dtype=np.float32),"
                "'right_arm_joint_state':np.zeros(6,dtype=np.float32),"
                "'right_ee_joint_state':np.zeros(1,dtype=np.float32)}}; "
                "c.call(func_name='update_obs',obs=obs); "
                "a=c.call(func_name='get_action'); "
                "print('HEALTH_OK', list(a[0].keys()) if a else a)"
                "\" 2>&1 | tail -15",
                timeout=50,
                label="E. Health + current tunnel URL",
            )

    finally:
        await dsw.close()
        print("\n[DONE] Disconnected.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
