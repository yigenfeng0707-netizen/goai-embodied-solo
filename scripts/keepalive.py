"""DSW instance idle-stop keepalive (REST-API heartbeat, local resident).

DSW idle auto-stop is based on gateway inactivity (~minutes). This periodically
calls an innocuous contents-listing via dswhub to keep the gateway active,
fighting idle auto-stop. Cannot fight hard runtime caps / manual / quota stops.

Usage:
    python keepalive.py                      # default every 240s, infinite
    python keepalive.py --interval 180       # every 3 min
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dswhub as d  # noqa: E402


def heartbeat() -> str:
    c = d.Dswhub()
    c.contents("")
    return f"keepalive {time.time():.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=240, help="heartbeat interval seconds")
    ap.add_argument("--max", type=int, default=0, help="max beats, 0=infinite")
    args = ap.parse_args()

    print(f"[keepalive] start id={d.INSTANCE} interval={args.interval}s max={args.max or 'inf'}", flush=True)
    n = 0
    try:
        while True:
            n += 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                out = heartbeat()
                print(f"[{ts}] #{n} OK: {out.strip()[:80]}", flush=True)
            except Exception as e:
                print(f"[{ts}] #{n} FAIL: {repr(e)[:160]}", flush=True)
            if args.max and n >= args.max:
                print(f"[keepalive] reached max {args.max}, exit", flush=True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[keepalive] Ctrl+C, exit")


if __name__ == "__main__":
    main()
