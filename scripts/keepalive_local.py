"""Local background keepalive for the DSW instance via the CDP/RawDSW channel.

The JupyterHub REST cookie is currently expired, so we keep the gateway active
by periodically executing a harmless command through the lab-page kernel
(RawDSW). This resets DSW's idle-stop timer. It only fights idle-stop, not
hard runtime caps / manual / quota stops.

Requires the lab page for the instance to remain open in the remote-debugging
Chrome (127.0.0.1:9222). Training itself runs detached and is unaffected.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _dsw_raw import RawDSW  # noqa: E402

INSTANCE = "dsw-2098907"

LOG = os.path.join(HERE, "keepalive_local.log")


def beat():
    d = RawDSW()
    try:
        out = d.run_python("import time; print('ka', round(time.time()))", timeout=30)
    finally:
        d.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] keepalive start id={INSTANCE} interval={args.interval}s max={args.max or 'inf'}\n")
    n = 0
    while True:
        n += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            out = beat()
            line = f"[{ts}] #{n} OK: {out.strip()[:70]}\n"
        except Exception as e:
            line = f"[{ts}] #{n} FAIL: {repr(e)[:160]}\n"
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line)
        if args.max and n >= args.max:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] reached max {args.max}, exit\n")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
