"""Corrects plans from the editor's queue with Claude Code or ChatGPT (Codex CLI) on this computer.

The editor cannot reach this computer, so this script goes to the editor: it asks for the next plan in the
"Corrigir com Claude" / "Corrigir com ChatGPT" queue, downloads the same package as "Pedir ajuda à IA", runs
`claude -p` or `codex exec` on it in a folder of its own, checks the answer (verificar.py, written next to the
package) and posts the correction back. The editor loads it into the plan as a draft when the reviewer opens
it; nothing is marked corrected.

    set EDITOR_TOKEN=...            (the editor password; never passed on the command line)
    python scripts/claude_worker.py                          # Claude jobs, until Ctrl+C
    python scripts/claude_worker.py --engines claude,codex   # Claude and ChatGPT jobs
    python scripts/claude_worker.py --engines openai         # GPT by the OpenAI API (OPENAI_API_KEY, paid per token)
    python scripts/claude_worker.py --engines anthropic      # Claude by the Anthropic API (CLAUDE_WORKER_API_KEY)
    python scripts/claude_worker.py --once                   # one plan, then exits
    python scripts/claude_worker.py --model sonnet           # passed to claude --model (--codex-model for codex)

Each job keeps its folder (package, correcao.json, <engine>.jsonl with the whole session) under --workdir,
outside the repository so the session does not pick up the project's CLAUDE.md / AGENTS.md.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

EDITOR = "https://botai-wallextractor-editor.uuclvw.easypanel.host"

PROMPT = """Você está rodando sozinho, sem ninguém para responder perguntas, na pasta de uma planta baixa
exportada pelo editor do WallExtractor. Leia o LEIA-ME.md e siga-o: corrija as paredes, portas e janelas de
planta.json comparando com planta.png e grave o resultado em correcao.json, nesta pasta, com a lista COMPLETA
de walls e openings, em pixels de planta.png, plan_id "{pid}".

Como trabalhar:
- Olhe planta.png e planta_numerada.png inteiras primeiro; leia as coordenadas nos recortes/ (a grade traz os
  valores originais).
- Pode escrever e rodar scripts Python nesta pasta (Pillow e numpy instalados) para juntar trechos alinhados,
  dividir nos encontros e cortar nos vãos.
- Antes de terminar, rode `python verificar.py correcao.json` e corrija até não sobrar nenhum ERRO (paredes
  passando por cima de vãos, paredes minúsculas, aberturas fora de qualquer parede...). Depois rode
  `python conferir.py correcao.json`, olhe conferencia.png e compare com planta.png nos pontos duvidosos.
- Não faça perguntas: decida pelo desenho. Dúvidas que restarem vão no campo "notas" do correcao.json.
- A última mensagem é um resumo de no máximo 3 linhas, em português, para o revisor: o que mudou em relação ao
  rascunho e o que ficou em dúvida.
"""

VERIFY = r'''"""Checks a correcao.json against the rules of LEIA-ME.md. Usage: python verificar.py correcao.json
Prints one line per problem; ERRO lines must be fixed, AVISO lines are worth a look. Exit code 1 on any ERRO."""
import json, math, sys
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = sys.argv[1] if len(sys.argv) > 1 else "correcao.json"
plan = json.load(open(path, encoding="utf-8"))
W, H = Image.open("planta.png").size
errs, warns = [], []
walls, ops = plan.get("walls", []), plan.get("openings", [])
ids = [x.get("id") for x in walls + ops]
for i in set(ids):
    if ids.count(i) > 1:
        errs.append(f"id repetido: {i}")


def pt(p):
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)


ok_w = []
for w in walls:
    if not (pt(w.get("start")) and pt(w.get("end")) and isinstance(w.get("thickness"), (int, float))):
        errs.append(f"{w.get('id')}: parede sem start/end/thickness numéricos")
    else:
        ok_w.append(w)
ok_o = []
for o in ops:
    if o.get("type") not in ("door", "window"):
        errs.append(f"{o.get('id')}: type deve ser door ou window")
    if not (pt(o.get("start")) and pt(o.get("end"))):
        errs.append(f"{o.get('id')}: abertura sem start/end numéricos")
    else:
        ok_o.append(o)
for x in ok_w + ok_o:
    for p in (x["start"], x["end"]):
        if not (-2 <= p[0] <= W + 2 and -2 <= p[1] <= H + 2):
            errs.append(f"{x['id']}: ponto {p} fora da imagem ({W} x {H})")

ts = sorted(w["thickness"] for w in ok_w)
t = ts[len(ts) // 2] if ts else 8.0


def geom(x):
    (x1, y1), (x2, y2) = x["start"], x["end"]
    L = math.hypot(x2 - x1, y2 - y1) or 1e-9
    return x1, y1, (x2 - x1) / L, (y2 - y1) / L, L


def skew(x):
    a = math.degrees(math.atan2(x["end"][1] - x["start"][1], x["end"][0] - x["start"][0]))
    return ((a % 90) + 135) % 90 - 45


for w in ok_w:
    L = geom(w)[4]
    if L < max(0.6 * t, 5):
        # a short stub between an opening and a corner is real; anywhere else it is a leftover piece
        jamb = any(math.dist(p, q) <= t / 2 for p in (w["start"], w["end"]) for o in ok_o for q in (o["start"], o["end"]))
        (warns if jamb else errs).append(f"{w['id']}: parede minúscula ({L:.0f} px, espessura típica {t:.0f} px)"
                                         + ("; confira se o toco ao lado da abertura existe" if jamb else "; junte ao vizinho ou apague"))
    if 1 <= abs(skew(w)) < 10:
        warns.append(f"{w['id']}: parede torta ({skew(w):.1f}°); se a planta não é inclinada, endireite")
    if w["thickness"] < 0.4 * t or w["thickness"] > 2.5 * t:
        warns.append(f"{w['id']}: espessura {w['thickness']:.0f} px muito longe da típica ({t:.0f} px)")

for o in ok_o:
    ox, oy, ux, uy, OL = geom(o)
    if OL < max(0.8 * t, 6):
        errs.append(f"{o['id']}: abertura minúscula ({OL:.0f} px)")
    near = False
    for w in ok_w:
        wx, wy, vx, vy, WL = geom(w)
        if abs(ux * vy - uy * vx) > math.sin(math.radians(8)):
            continue  # not parallel
        mx, my = ox + ux * OL / 2, oy + uy * OL / 2
        if abs(-(mx - wx) * vy + (my - wy) * vx) > max(t * 0.75, 4):
            continue  # not on the same line
        a = sorted(((o["start"][0] - wx) * vx + (o["start"][1] - wy) * vy, (o["end"][0] - wx) * vx + (o["end"][1] - wy) * vy))
        over = min(a[1], WL) - max(a[0], 0)
        if over > max(0.25 * OL, 4):
            errs.append(f"{o['id']} ({o.get('type')}): a parede {w['id']} passa por cima do vão ({over:.0f} px); "
                        f"corte a parede nas bordas da abertura")
        if over > -max(t, 6):
            near = True
    if not near:
        errs.append(f"{o['id']}: abertura sem parede alinhada encostando nela (abertura solta ou fora do eixo)")
    if 1 <= abs(skew(o)) < 10:
        warns.append(f"{o['id']}: abertura torta ({skew(o):.1f}°)")

for e in errs:
    print("ERRO", e)
for w in warns:
    print("AVISO", w)
print(f"{len(walls)} paredes, {len(ops)} aberturas: {len(errs)} erro(s), {len(warns)} aviso(s)")
sys.exit(1 if errs else 0)
'''


class Editor:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def call(self, path: str, body=None, raw: bool = False, timeout: int = 120):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + "/api" + path, data=data, method="POST" if data is not None else "GET",
                                     headers={"X-Token": self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
        return payload if raw else json.loads(payload)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def describe(event: dict) -> str | None:
    """A short Portuguese line for the last tool Claude used, shown in the editor as progress."""
    if event.get("type") != "assistant":
        return None
    for c in event.get("message", {}).get("content", []):
        if c.get("type") != "tool_use":
            continue
        i = c.get("input", {})
        name = c.get("name")
        if name == "Read":
            return "olhando " + Path(i.get("file_path", "")).name
        if name == "Bash":
            return "rodando " + " ".join(str(i.get("command", "")).split())[:70]
        if name in ("Write", "Edit"):
            return "escrevendo " + Path(i.get("file_path", "")).name
        return name
    return None


def describe_codex(event: dict) -> str | None:
    """The same for `codex exec --json` events."""
    if event.get("type") != "item.started":
        return None
    it = event.get("item", {})
    kind = it.get("type")
    if kind == "command_execution":
        c = str(it.get("command", ""))
        m = re.search(r"-Command\s+(.*)", c, re.S)  # Windows: the command comes wrapped in pwsh.exe -Command '...'
        return "rodando " + " ".join((m.group(1).strip("'\"") if m else c).split())[:70]
    if kind == "file_change":
        return "escrevendo " + ", ".join(Path(c.get("path", "")).name for c in it.get("changes", []))[:70]
    if kind in ("reasoning", "agent_message", None):
        return None
    return kind.replace("_", " ")


DOTENV = Path(__file__).resolve().parent.parent / ".env"  # repository root; in .gitignore


def dotenv(name: str) -> str:
    """NAME=value from the repository's .env (blank lines and # comments skipped, optional quotes)."""
    if not DOTENV.is_file():
        return ""
    for line in DOTENV.read_text(encoding="utf-8-sig").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#") and k.strip().removeprefix("export ").strip() == name:
            return v.strip().strip("'\"")
    return ""


def user_env(name: str) -> str:
    """A variable from the process environment, the repository's .env or, on Windows, the user's saved
    variables (a terminal opened before the variable was created, or inside an app started before, lacks it)."""
    v = os.environ.get(name, "") or dotenv(name)
    if v or os.name != "nt":
        return v
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0])
    except OSError:
        return ""


def find_codex() -> str:
    """The Codex desktop app ships a newer CLI than an old global npm install; prefer the newest of the two."""
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI" / "Codex" / "bin"
    app = sorted(base.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime) if base.is_dir() else []
    return str(app[-1]) if app else (shutil.which("codex") or "codex")


LIMIT = re.compile(r"usage limit|hit your (usage )?limit|limit reached|rate.?limit|quota", re.I)
NAMES = {"claude": "Claude", "codex": "ChatGPT", "openai": "GPT (API)", "anthropic": "Claude (API)"}
# US$ per million tokens (input, cache read, output) for Claude through the API. Cache writes cost 1.25x input
# (5 min) or 2x (1 h). The CLI's own total_cost_usd misprices models newer than itself.
CLAUDE_PRICES = {"claude-sonnet-5-5": (2.0, 0.20, 10.0), "claude-opus-5-5": (4.0, 0.20, 20.0),
                 "claude-sonnet-5": (2.0, 0.20, 10.0), "claude-haiku-4-5": (1.0, 0.10, 5.0)}


def claude_cost(usage: dict, model: str):
    p = CLAUDE_PRICES.get(model)
    if not p or not usage:
        return None
    pi, pr, po = p
    cc = usage.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens", 0) or 0
    w5m = cc.get("ephemeral_5m_input_tokens", usage.get("cache_creation_input_tokens", 0) - w1h) or 0
    return (usage.get("input_tokens", 0) * pi + w5m * 1.25 * pi + w1h * 2 * pi
            + usage.get("cache_read_input_tokens", 0) * pr + usage.get("output_tokens", 0) * po) / 1e6
# US$ per million tokens (input, cached input, output) for the cost line of API jobs
PRICES = {"gpt-5.4-mini": (0.75, 0.075, 4.50), "gpt-5.4-nano": (0.20, 0.02, 1.25), "gpt-5.4": (2.50, 0.25, 15.0),
          "gpt-6-luna": (0.10, 0.01, 0.50), "gpt-6-sol": (2.00, 0.20, 10.0), "gpt-6-astra": (10.0, 1.0, 50.0)}


def limit_until(text: str) -> float | None:
    """When the session stopped on the plan's usage limit: the time it comes back ("try again at 2:09 PM",
    "resets 3pm", "...|1760000000"), or 30 min from now when the message has no time. None otherwise."""
    if not text or not LIMIT.search(text):
        return None
    now = time.time()
    m = re.search(r"\|(\d{10})\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:at|resets?(?: at)?)\s+(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", text, re.I)
    if m:
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        ap = (m.group(3) or "").lower().replace(".", "")
        if ap == "pm" and h < 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        t = time.localtime(now)
        at = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, h, mi, 0, 0, 0, -1))
        return at if at > now else at + 86400
    return now + 1800


def find_json(text: str):
    """correcao.json missing: the JSON in the last message, if any."""
    for m in reversed(list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S))):
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    return None


def run_job(ed: Editor, job: dict, args) -> None:
    jid, pid = job["id"], job["pid"]
    folder = Path(args.workdir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{pid}"
    folder.mkdir(parents=True, exist_ok=True)
    log(f"job {jid}: planta {pid} ({job.get('title')}) com {job.get('engine') or 'claude'}"
        f"{' ' + job['model'] if job.get('model') else ''} em {folder}")
    ed.call(f"/ai/jobs/{jid}", {"message": "baixando o pacote"})
    zipfile.ZipFile(io.BytesIO(ed.call(f"/plans/{pid}/ai", raw=True, timeout=300))).extractall(folder)
    (folder / "verificar.py").write_text(VERIFY, encoding="utf-8")

    engine = job.get("engine") or "claude"
    model = job.get("model") or ""
    if engine in ("codex", "openai"):
        # workspace-write: it writes only in the job folder; no network for its commands. The two whole images
        # go with the prompt; the zoomed tiles it opens itself.
        # the prompt goes through stdin: -i takes every argument after it as an image
        cmd = [args.codex, "exec", "--json", "--sandbox", "workspace-write", "--skip-git-repo-check", "-C", str(folder),
               "-o", str(folder / "ultima_mensagem.txt")]
        if model or args.codex_model:
            cmd += ["-m", model or args.codex_model]
        cmd += ["-i", str(folder / "planta.png"), str(folder / "planta_numerada.png")]
    else:
        cmd = [args.claude, "-p", PROMPT.format(pid=pid), "--output-format", "stream-json", "--verbose",
               "--permission-mode", "acceptEdits",
               "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Bash(python *)", "Bash(python3 *)", "Bash(ls *)",
               "--disallowedTools", "WebFetch", "WebSearch", "Agent", "Task"]
        if model or args.model:
            cmd += ["--model", model or args.model]
    env = {k: v for k, v in os.environ.items() if k != "EDITOR_TOKEN"}  # the session never sees the password
    env["PYTHONUTF8"] = "1"  # Windows: the session's Python scripts print accents without cp1252 errors
    env.pop("CODEX_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)  # subscription jobs stay on the subscription
    if engine == "anthropic":  # this run only: billed to the API key; the Claude Code login is left as it is
        key = user_env("CLAUDE_WORKER_API_KEY")
        if not key:
            raise RuntimeError("CLAUDE_WORKER_API_KEY não está definida neste computador")
        env["ANTHROPIC_API_KEY"] = key
    if engine == "openai":  # this run only: API-key auth, billed per token; the ChatGPT login is left as it is
        key = user_env("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não está definida neste computador")
        env["CODEX_API_KEY"] = key
    codex = engine in ("codex", "openai")
    proc = subprocess.Popen(cmd, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if codex else subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", env=env)
    if codex:
        proc.stdin.write(PROMPT.format(pid=pid))
        proc.stdin.close()
    state = {"last": "começando", "steps": 0, "result": None, "stop": None, "usage": {}, "fail": None}

    def reader():
        with open(folder / f"{engine}.jsonl", "w", encoding="utf-8") as out:
            for line in proc.stdout:
                out.write(line)
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                d = describe_codex(ev) if codex else describe(ev)
                if d:
                    state["last"], state["steps"] = d, state["steps"] + 1
                if ev.get("type") == "result":
                    state["result"] = ev
                elif ev.get("type") == "turn.completed":
                    for k, v in (ev.get("usage") or {}).items():
                        state["usage"][k] = state["usage"].get(k, 0) + (v or 0)
                elif ev.get("type") in ("turn.failed", "error"):
                    state["fail"] = str((ev.get("error") or {}).get("message") or ev.get("message"))[:300]

    errlines: list = []
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    th_err = threading.Thread(target=lambda: errlines.extend(proc.stderr), daemon=True)  # a full pipe would block claude
    th_err.start()
    t0 = beat = time.time()
    while proc.poll() is None:
        time.sleep(1)
        el = time.time() - t0
        if el > args.timeout_min * 60:
            state["stop"] = f"passou de {args.timeout_min} min"
        elif time.time() - beat >= 20:
            beat = time.time()
            try:
                j = ed.call(f"/ai/jobs/{jid}", {"message": f"{el / 60:.0f} min, {state['steps']} passos; {state['last']}"})
                if j.get("status") == "cancelled":
                    state["stop"] = "cancelado no editor"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                log(f"  aviso: editor não respondeu ({exc.__class__.__name__})")
        if state["stop"]:
            proc.kill()
            break
    proc.wait()
    th.join(timeout=10)
    minutes = (time.time() - t0) / 60
    if state["stop"] == "cancelado no editor":
        log(f"job {jid}: cancelado no editor")
        return None
    res = state["result"] or {}
    if codex:
        last = folder / "ultima_mensagem.txt"
        res = {"result": last.read_text(encoding="utf-8", errors="replace") if last.is_file() else (state["fail"] or "")}
    if state["stop"]:
        ed.call(f"/ai/jobs/{jid}", {"status": "error", "message": f"parado: {state['stop']}; pasta {folder}"})
        log(f"job {jid}: {state['stop']}")
        return None
    th_err.join(timeout=5)
    until = limit_until(" ".join([state["fail"] or "", str(res.get("result") or ""), "".join(errlines)[-2000:]]))
    if until and not (folder / "correcao.json").is_file():
        back = time.strftime("%H:%M", time.localtime(until))
        ed.call(f"/ai/jobs/{jid}", {"status": "queued",
                                    "message": f"limite de uso do {NAMES.get(engine, engine)} até {back}; a planta voltou para a fila"})
        log(f"job {jid}: limite de uso do {engine} até {back}; planta devolvida à fila")
        return (engine, until)

    corr = None
    if (folder / "correcao.json").is_file():
        try:
            corr = json.loads((folder / "correcao.json").read_text(encoding="utf-8"))
        except ValueError as exc:
            corr = None
            log(f"  correcao.json inválido: {exc}")
    corr = corr or find_json(res.get("result", ""))
    if not isinstance(corr, dict) or not isinstance(corr.get("walls"), list):
        err = "".join(errlines)[-300:]
        why = res.get("result") or err or f"saída {proc.returncode}"
        ed.call(f"/ai/jobs/{jid}", {"status": "error", "message": f"sem correcao.json ({' '.join(str(why).split())[:300]}); pasta {folder}"})
        log(f"job {jid}: sem correcao.json")
        return None
    corr.setdefault("openings", [])
    (folder / "_final.json").write_text(json.dumps(corr, ensure_ascii=False), encoding="utf-8")
    chk = subprocess.run([sys.executable, "verificar.py", "_final.json"], cwd=folder, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    errs = [ln[5:] for ln in chk.stdout.splitlines() if ln.startswith("ERRO ")]
    cost = res.get("total_cost_usd")
    u = state["usage"]
    tokens = f", {(u.get('input_tokens', 0) + u.get('output_tokens', 0)) / 1000:.0f} mil tokens" if u else ""
    if engine == "anthropic":
        c2 = claude_cost(res.get("usage") or {}, model)
        cost = c2 if c2 is not None else cost
    if engine == "openai" and u and model in PRICES:
        pi, pc, po = PRICES[model]
        cached = u.get("cached_input_tokens", 0)
        cost = ((u.get("input_tokens", 0) - cached) * pi + cached * pc + u.get("output_tokens", 0) * po) / 1e6
    summary = " ".join(str(res.get("result") or "").split())[:600]
    msg = (f"{len(corr['walls'])} paredes, {len(corr['openings'])} aberturas em {minutes:.0f} min, {state['steps']} passos"
           + (f", custo estimado US$ {cost:.2f}" if isinstance(cost, (int, float)) else "") + tokens
           + (f". Sobraram {len(errs)} erro(s) da verificação: {'; '.join(errs[:4])}" if errs else ". Verificação sem erros")
           + (f". Resumo: {summary}" if summary else ""))
    ed.call(f"/ai/jobs/{jid}", {"status": "done", "result": corr, "message": msg, "minutes": round(minutes, 1),
                                "cost_usd": round(cost, 4) if isinstance(cost, (int, float)) else None,
                                "tokens": (u.get("input_tokens", 0) + u.get("output_tokens", 0)) if u else None})
    log(f"job {jid}: pronto. {msg[:200]}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", default=os.environ.get("EDITOR_URL", EDITOR))
    ap.add_argument("--name", default=socket.gethostname(), help="nome mostrado no editor")
    ap.add_argument("--workdir", default=str(Path.home() / ".wallextractor" / "claude_jobs"))
    ap.add_argument("--engines", default="claude", help="quais filas atender: claude, codex ou claude,codex")
    ap.add_argument("--claude", default=shutil.which("claude") or "claude")
    ap.add_argument("--codex", default=find_codex())
    ap.add_argument("--model", default="", help="repassado a claude --model (padrão: o do Claude Code)")
    ap.add_argument("--codex-model", default="", help="repassado a codex exec -m (padrão: o do Codex)")
    ap.add_argument("--timeout-min", type=float, default=45)
    ap.add_argument("--poll", type=float, default=15, help="segundos entre pedidos de trabalho")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    token = user_env("EDITOR_TOKEN")
    if not token:
        sys.exit("defina EDITOR_TOKEN (a senha do editor) no ambiente")
    ed = Editor(args.editor, token)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    log(f"worker {args.name} ({', '.join(engines)}) pedindo trabalho a {args.editor} (Ctrl+C para sair)")
    paused: dict = {}  # engine -> time its usage limit ends
    while True:
        now = time.time()
        for e in [e for e, t in paused.items() if t <= now]:
            log(f"{e}: limite de uso terminou, voltando a pedir trabalho")
            del paused[e]
        active = [e for e in engines if e not in paused]
        if not active:
            if args.once:
                return
            time.sleep(min(60.0, max(5.0, min(paused.values()) - now)))
            continue
        try:
            job = ed.call("/ai/jobs/claim", {"worker": args.name, "engines": active})["job"]
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log(f"editor fora do ar ({exc.__class__.__name__}); tento de novo")
            time.sleep(60)
            continue
        if job:
            try:
                hit = run_job(ed, job, args)
                if hit:
                    paused[hit[0]] = hit[1]
                    log(f"{hit[0]}: pausado até {time.strftime('%H:%M', time.localtime(hit[1]))}")
            except Exception as exc:  # noqa: BLE001 - one bad job must not stop the worker
                log(f"job {job['id']}: falhou ({exc.__class__.__name__}: {exc})")
                try:
                    ed.call(f"/ai/jobs/{job['id']}", {"status": "error", "message": f"{exc.__class__.__name__}: {exc}"[:500]})
                except Exception:  # noqa: BLE001
                    pass
            if args.once:
                return
        elif args.once:
            log("fila vazia")
            return
        else:
            time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
