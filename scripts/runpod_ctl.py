"""Control a RunPod training pod over HTTPS only.

The environment this runs from may not allow SSH, so the pod is driven with
the RunPod REST API (create/stop/terminate) and the JupyterLab contents API
exposed by the RunPod PyTorch template on ``https://<pod>-8888.proxy.runpod.net``.

Commands:
  gpus                         list GPU types with price and stock
  create [--gpu ...] [--branch B] [--train N --val M --epochs E ...]
  status                       pod status
  log [--tail N]               print /workspace/job.log
  get <remote> [<local>]       download a file from /workspace (text or binary)
  cmd "<shell>"                run a shell snippet in the pod (see runpod_job.sh loop)
  cmdlog                       print /workspace/cmd.log
  stop | terminate

State (pod id, token) is kept in --state (default ~/.wallextractor_pod.json).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:1.3.3-rc.171-cu1281-torch280-ubuntu2204"
DEFAULT_GPUS = ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A4500", "NVIDIA RTX A4000", "NVIDIA RTX 4000 Ada Generation",
                "NVIDIA GeForce RTX 4090", "NVIDIA A40"]


def _key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        sys.exit("RUNPOD_API_KEY is not set")
    return k


def _http(method: str, url: str, body=None, headers=None, timeout=60):
    data = None
    hdrs = {"Accept": "application/json", "User-Agent": "wallextractor-runpod-ctl/0.1"}
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def rest(method: str, path: str, body=None):
    status, raw = _http(method, REST + path, body, {"Authorization": f"Bearer {_key()}"})
    try:
        return status, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return status, raw.decode(errors="replace")


def gql(query: str):
    status, raw = _http("POST", GQL, {"query": query}, {"Authorization": f"Bearer {_key()}"})
    return json.loads(raw)


def load_state(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(path: str, st: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)


# ---------------------------------------------------------------- jupyter
def jup_url(st: dict, path: str) -> str:
    return f"https://{st['pod_id']}-8888.proxy.runpod.net/api/contents/{urllib.parse.quote(path)}"


def jup_get(st: dict, path: str, binary: bool = False):
    fmt = "base64" if binary else "text"
    for prefix in ("workspace/", ""):
        url = jup_url(st, prefix + path) + f"?type=file&format={fmt}&content=1"
        status, raw = _http("GET", url, headers={"Authorization": f"token {st['token']}"}, timeout=120)
        if status == 200:
            d = json.loads(raw)
            content = d.get("content") or ""
            return base64.b64decode(content) if binary else content
        if status not in (404,):
            return None
    return None


def jup_put(st: dict, path: str, text: str) -> bool:
    body = {"type": "file", "format": "text", "content": text}
    for prefix in ("workspace/", ""):
        status, raw = _http("PUT", jup_url(st, prefix + path), body, {"Authorization": f"token {st['token']}"})
        if status in (200, 201):
            return True
    return False


# ---------------------------------------------------------------- commands
def cmd_gpus(args, st):
    d = gql("{ gpuTypes { id displayName memoryInGb lowestPrice(input:{gpuCount:1}) { uninterruptablePrice stockStatus } } }")
    rows = []
    for g in d["data"]["gpuTypes"]:
        lp = g.get("lowestPrice") or {}
        if lp.get("uninterruptablePrice"):
            rows.append((lp["uninterruptablePrice"], g["id"], g["memoryInGb"], lp.get("stockStatus")))
    for price, gid, mem, stock in sorted(rows):
        print(f"${price:.2f}/h  {mem:>4} GB  {stock:<7} {gid}")


def cmd_create(args, st):
    token = secrets.token_urlsafe(24)
    env = {
        "JUPYTER_PASSWORD": token,
        "WE_BRANCH": args.branch,
        "WE_TRAIN": str(args.train), "WE_VAL": str(args.val), "WE_EPOCHS": str(args.epochs),
        "WE_SIZE": str(args.size), "WE_BATCH": str(args.batch), "WE_MODEL": args.model,
        "WE_MAX_MINUTES": str(args.max_minutes), "WE_IDLE_MINUTES": str(args.idle_minutes),
        "WE_HARD_MINUTES": str(args.hard_minutes),
        "RUNPOD_API_KEY": _key(),
    }
    # The template's /start.sh brings up Jupyter; our job runs in the background and logs to a file.
    bootstrap = (
        "mkdir -p /workspace && (curl -fsSL https://raw.githubusercontent.com/maxrosan/WallExtractor/"
        f"{args.branch}/scripts/runpod_job.sh -o /workspace/job.sh && bash /workspace/job.sh) "
        "> /workspace/job.log 2>&1 & exec /start.sh"
    )
    body = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": args.gpu or DEFAULT_GPUS,
        "gpuCount": 1,
        "cloudType": args.cloud,
        "interruptible": False,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "ports": ["8888/http"],
        "env": env,
        "dockerStartCmd": ["bash", "-c", bootstrap],
    }
    status, d = rest("POST", "/pods", body)
    if status not in (200, 201):
        sys.exit(f"create failed: HTTP {status} {d}")
    st.update({"pod_id": d["id"], "token": token, "created": time.time(), "gpu": d.get("machine", {}).get("gpuTypeId")
               or d.get("gpuTypeId"), "cost_per_hr": d.get("costPerHr")})
    save_state(args.state, st)
    print(json.dumps({k: d.get(k) for k in ("id", "desiredStatus", "costPerHr", "machineId", "gpuTypeId")}))


def cmd_status(args, st):
    if not st.get("pod_id"):
        sys.exit("no pod in state")
    status, d = rest("GET", f"/pods/{st['pod_id']}")
    if status != 200:
        print(f"HTTP {status}: {d}")
        return
    keys = ("id", "name", "desiredStatus", "costPerHr", "gpuTypeId", "lastStatusChange", "uptimeSeconds")
    out = {k: d.get(k) for k in keys}
    if isinstance(d.get("runtime"), dict):
        out["uptimeSeconds"] = d["runtime"].get("uptimeInSeconds")
    elapsed_h = (time.time() - st.get("created", time.time())) / 3600
    out["est_cost_usd"] = round(elapsed_h * float(d.get("costPerHr") or st.get("cost_per_hr") or 0), 3)
    print(json.dumps(out))


def cmd_log(args, st):
    txt = jup_get(st, "job.log")
    if txt is None:
        print("(log not reachable yet)")
        return
    lines = txt.splitlines()
    print("\n".join(lines[-args.tail:]) if args.tail else txt)


def cmd_cmdlog(args, st):
    txt = jup_get(st, "cmd.log")
    print(txt if txt is not None else "(no cmd.log)")


def cmd_get(args, st):
    binary = not args.remote.endswith((".txt", ".log", ".json", ".jsonl", ".csv", ".md"))
    data = jup_get(st, args.remote, binary=binary)
    if data is None:
        sys.exit("not found")
    local = args.local or os.path.basename(args.remote)
    with open(local, "wb" if binary else "w", encoding=None if binary else "utf-8") as f:
        f.write(data)
    print(f"saved {local} ({len(data)} bytes)")


def cmd_cmd(args, st):
    ok = jup_put(st, "cmd.sh", args.shell + "\n")
    print("queued" if ok else "upload failed")


def cmd_stop(args, st):
    print(rest("POST", f"/pods/{st['pod_id']}/stop"))


def cmd_terminate(args, st):
    print(rest("DELETE", f"/pods/{st['pod_id']}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=os.path.expanduser("~/.wallextractor_pod.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    c = sub.add_parser("create")
    c.add_argument("--name", default="wallextractor-seg")
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--gpu", action="append")
    c.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    c.add_argument("--disk", type=int, default=30)
    c.add_argument("--branch", default="main")
    c.add_argument("--train", type=int, default=400)
    c.add_argument("--val", type=int, default=100)
    c.add_argument("--epochs", type=int, default=15)
    c.add_argument("--size", type=int, default=512)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--model", default="nvidia/mit-b1")
    c.add_argument("--max-minutes", type=int, default=45)
    c.add_argument("--idle-minutes", type=int, default=30)
    c.add_argument("--hard-minutes", type=int, default=150)
    sub.add_parser("status")
    lg = sub.add_parser("log")
    lg.add_argument("--tail", type=int, default=60)
    sub.add_parser("cmdlog")
    g = sub.add_parser("get")
    g.add_argument("remote")
    g.add_argument("local", nargs="?")
    cm = sub.add_parser("cmd")
    cm.add_argument("shell")
    sub.add_parser("stop")
    sub.add_parser("terminate")
    args = ap.parse_args()
    st = load_state(args.state)
    {"gpus": cmd_gpus, "create": cmd_create, "status": cmd_status, "log": cmd_log, "cmdlog": cmd_cmdlog,
     "get": cmd_get, "cmd": cmd_cmd, "stop": cmd_stop, "terminate": cmd_terminate}[args.cmd](args, st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
