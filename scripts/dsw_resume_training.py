"""One-command resume of RoboDojo ACT training on a NEW DSW instance.

Prep so that after the current instance stops, resuming later is a single call:

    python dsw_resume_training.py "<new dsw url>"

It will:
  1. Parse dsw-<id> from the URL and patch dswhub.INSTANCE (so dswhub,
     dsw_refresh_cookie, keepalive and ckpt-watcher all target the new instance).
  2. Refresh the gateway session cookie via Chrome CDP (dsw_refresh_cookie.main),
     which restores the user's Chrome afterwards.
  3. Verify connectivity (dswhub.me()).
  4. Launch /mnt/data/persist/run_train.sh on the new instance (detached,
     logging to /root/train_cont.log). run_train.sh auto-restores code+dataset
     from NAS and auto-resumes from the latest checkpoint in ckpt_dir.
  5. (Re)start the local keepalive + ckpt-watcher so monitoring resumes.

Note: requires the user's Chrome to be logged into Aliyun/ModelScope for the
cookie refresh. The training itself reads checkpoints from durable NAS, so
progress is never lost across instance rebuilds.
"""
from __future__ import annotations
import os, sys, re, subprocess, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse_instance(url: str) -> str:
    m = re.search(r"dsw-([A-Za-z0-9_-]+)", url)
    if not m:
        raise SystemExit("ERROR: cannot find dsw-<id> in the provided URL")
    return m.group(1)


def set_instance(inst_id: str):
    """Patch dswhub.py INSTANCE and reload it in this process."""
    p = os.path.join(HERE, "dswhub.py")
    s = open(p, encoding="utf-8").read()
    s2 = re.sub(r'INSTANCE = "[^"]*"', f'INSTANCE = "{inst_id}"', s)
    assert f'INSTANCE = "{inst_id}"' in s2, "failed to patch INSTANCE"
    open(p, "w", encoding="utf-8").write(s2)
    import dswhub
    importlib.reload(dswhub)
    dswhub.INSTANCE = inst_id
    return dswhub


def refresh_cookie(d):
    import dsw_refresh_cookie
    importlib.reload(dsw_refresh_cookie)
    dsw_refresh_cookie.main()


def verify(d):
    return d.Dswhub().me()


def launch_training(d):
    SCRIPT = r'''
import subprocess, os
log = "/root/train_cont.log"
open(log, "w").close()
p = subprocess.Popen(["bash", "/mnt/data/persist/run_train.sh"],
    stdout=open(log, "a"), stderr=subprocess.STDOUT,
    env=os.environ, start_new_session=True)
print("LAUNCHED training pid", p.pid, "log", log)
'''
    return d.Dswhub().run_python(SCRIPT, timeout=60)


def restart_local_watchers():
    """Kill any running keepalive/ckpt-watcher and start fresh (so they pick
    up the new dswhub.INSTANCE)."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline"],
            capture_output=True, text=True).stdout
    except Exception:
        out = ""
    kill_pids = []
    for line in out.splitlines():
        if "keepalive.py" in line or "_dsw_ckptwatch.py" in line:
            m = re.search(r"(\d+)\s*$", line.strip())
            if m:
                kill_pids.append(m.group(1))
    for pid in kill_pids:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, text=True)
    if kill_pids:
        print("killed old watchers:", kill_pids)
        time.sleep(2)
    for name in ["keepalive.py", "_dsw_ckptwatch.py"]:
        logp = os.path.join(HERE, name.replace(".py", ".resume.log"))
        f = open(logp, "a", encoding="utf-8")
        subprocess.Popen(["python", os.path.join(HERE, name)],
                         stdout=f, stderr=subprocess.STDOUT,
                         creationflags=0x00000008)  # DETACHED_PROCESS
        print("started", name, "->", logp)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python dsw_resume_training.py <dsw_url>")
    url = sys.argv[1]
    inst = parse_instance(url)
    print("[resume] target instance:", inst)
    d = set_instance(inst)
    refresh_cookie(d)
    info = verify(d)
    print("[resume] connected as:", info.get("identity"))
    launch_training(d)
    restart_local_watchers()
    print("[resume] DONE. Monitor via _dsw_ckptwatch.log ; training auto-resumes "
          "from the latest checkpoint in ckpt_dir.")


if __name__ == "__main__":
    import time
    main()
