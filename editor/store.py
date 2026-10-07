"""SQLite-backed store for the editor: one row per plan (a PDF page), files on disk.

Layout under ``EDITOR_DATA`` (default ``/data``):
  editor.sqlite            plans table
  pdfs/<id>.pdf            the uploaded PDF
  renders/<id>.png         base render of the plan region (long side 2000 px)

The ai_jobs table is the queue of plans waiting for a correction by an AI agent on the user's own
computer (scripts/claude_worker.py claims a job, runs Claude Code or the Codex CLI on it and posts the result
back); `engine` says which one (claude, codex = ChatGPT subscription, openai = OpenAI API through the
Codex CLI) and `model` which model, when not the engine's default.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

STATUSES = ("pending", "corrected", "skipped")
JOB_STATUSES = ("queued", "running", "done", "error", "cancelled")
ENGINES = ("claude", "codex", "openai")
JOB_STALE_S = 15 * 60  # a running job without news from its worker for this long is given up


class Store:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "pdfs"), exist_ok=True)
        os.makedirs(os.path.join(root, "renders"), exist_ok=True)
        self.path = os.path.join(root, "editor.sqlite")
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY, title TEXT, file TEXT, page INTEGER, status TEXT,
                    created REAL, updated REAL, machine TEXT, corrected TEXT, meta TEXT)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS ai_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, status TEXT, created REAL, updated REAL,
                    worker TEXT, message TEXT, result TEXT, applied INTEGER DEFAULT 0)"""
            )
            if "engine" not in [r[1] for r in c.execute("PRAGMA table_info(ai_jobs)")]:
                c.execute("ALTER TABLE ai_jobs ADD COLUMN engine TEXT DEFAULT 'claude'")
            if "model" not in [r[1] for r in c.execute("PRAGMA table_info(ai_jobs)")]:
                c.execute("ALTER TABLE ai_jobs ADD COLUMN model TEXT")
            if "cost_usd" not in [r[1] for r in c.execute("PRAGMA table_info(ai_jobs)")]:
                for col in ("cost_usd REAL", "tokens INTEGER", "minutes REAL"):
                    c.execute(f"ALTER TABLE ai_jobs ADD COLUMN {col}")
            # jobs finished before these columns: the numbers are in the message text
            for r in c.execute("SELECT id, message FROM ai_jobs WHERE status = 'done' AND minutes IS NULL").fetchall():
                m = r["message"] or ""
                cost = re.search(r"custo estimado US\$ (\d+(?:\.\d+)?)", m)
                tok = re.search(r"(\d+) mil tokens", m)
                mins = re.search(r" em (\d+) min", m)
                c.execute("UPDATE ai_jobs SET cost_usd = ?, tokens = ?, minutes = ? WHERE id = ?",
                          (float(cost.group(1)) if cost else None, int(tok.group(1)) * 1000 if tok else None,
                           float(mins.group(1)) if mins else -1.0, r["id"]))

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ------------------------------------------------------------ paths
    def pdf_path(self, pid: str) -> str:
        return os.path.join(self.root, "pdfs", f"{pid}.pdf")

    def render_path(self, pid: str) -> str:
        return os.path.join(self.root, "renders", f"{pid}.png")

    # ------------------------------------------------------------ rows
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:10]

    def create(self, pid: str, title: str, file: str, page: int, machine: dict, meta: dict) -> dict:
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO plans VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, title, file, page, "pending", now, now, json.dumps(machine, ensure_ascii=False), None,
                 json.dumps(meta, ensure_ascii=False)),
            )
        return self.get(pid)  # type: ignore[return-value]

    def list(self, status: Optional[str] = None) -> List[dict]:
        q = "SELECT id, title, file, page, status, created, updated, machine, corrected, meta FROM plans"
        args: tuple = ()
        if status:
            q += " WHERE status = ?"
            args = (status,)
        q += " ORDER BY created DESC"
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        out = []
        for r in rows:
            plan = json.loads(r["corrected"] or r["machine"])
            out.append({
                "id": r["id"], "title": r["title"], "file": r["file"], "page": r["page"], "status": r["status"],
                "created": r["created"], "updated": r["updated"], "has_correction": r["corrected"] is not None,
                "walls": len(plan.get("walls", [])), "openings": len(plan.get("openings", [])),
                "questions": len(plan.get("questions", [])),
            })
        return out

    def get(self, pid: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM plans WHERE id = ?", (pid,)).fetchone()
        if r is None:
            return None
        return {
            "id": r["id"], "title": r["title"], "file": r["file"], "page": r["page"], "status": r["status"],
            "created": r["created"], "updated": r["updated"],
            "machine": json.loads(r["machine"]),
            "corrected": json.loads(r["corrected"]) if r["corrected"] else None,
            "meta": json.loads(r["meta"] or "{}"),
        }

    def save_correction(self, pid: str, plan: Dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE plans SET corrected = ?, updated = ? WHERE id = ?",
                      (json.dumps(plan, ensure_ascii=False), time.time(), pid))

    def set_status(self, pid: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock, self._conn() as c:
            c.execute("UPDATE plans SET status = ?, updated = ? WHERE id = ?", (status, time.time(), pid))

    def set_title(self, pid: str, title: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE plans SET title = ?, updated = ? WHERE id = ?", (title, time.time(), pid))

    def delete(self, pid: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM plans WHERE id = ?", (pid,))
            c.execute("DELETE FROM ai_jobs WHERE pid = ?", (pid,))
        for p in (self.pdf_path(pid), self.render_path(pid)):
            if os.path.isfile(p):
                os.remove(p)

    # ------------------------------------------------------------ AI jobs
    @staticmethod
    def _job(r: sqlite3.Row, with_result: bool = False) -> dict:
        out = {k: r[k] for k in ("id", "pid", "status", "created", "updated", "worker", "message", "engine", "model",
                                         "cost_usd", "tokens", "minutes")}
        out["applied"] = bool(r["applied"])
        out["has_result"] = r["result"] is not None
        if with_result:
            out["result"] = json.loads(r["result"]) if r["result"] else None
        return out

    def _expire_stale(self, c: sqlite3.Connection) -> None:
        c.execute("UPDATE ai_jobs SET status = 'error', message = ?, updated = ? WHERE status = 'running' AND updated < ?",
                  ("o worker parou de dar notícias; peça de novo", time.time(), time.time() - JOB_STALE_S))

    def job_add(self, pid: str, engine: str = "claude", model: Optional[str] = None) -> Optional[dict]:
        """Queue a plan for `engine`; None when it is already queued or running (for any engine)."""
        now = time.time()
        with self._lock, self._conn() as c:
            self._expire_stale(c)
            if c.execute("SELECT 1 FROM ai_jobs WHERE pid = ? AND status IN ('queued', 'running')", (pid,)).fetchone():
                return None
            cur = c.execute("INSERT INTO ai_jobs (pid, status, created, updated, message, engine, model) "
                            "VALUES (?, 'queued', ?, ?, '', ?, ?)", (pid, now, now, engine, model))
            r = c.execute("SELECT * FROM ai_jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._job(r)

    def job_list(self, limit: int = 200) -> List[dict]:
        with self._lock, self._conn() as c:
            self._expire_stale(c)
            rows = c.execute("SELECT * FROM ai_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._job(r) for r in rows]

    def job_get(self, jid: int, with_result: bool = False) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM ai_jobs WHERE id = ?", (jid,)).fetchone()
        return self._job(r, with_result) if r else None

    def job_latest(self, pid: str) -> Optional[dict]:
        with self._lock, self._conn() as c:
            self._expire_stale(c)
            r = c.execute("SELECT * FROM ai_jobs WHERE pid = ? ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
        return self._job(r, with_result=True) if r else None

    def job_stats(self) -> List[dict]:
        """Per engine and model, over the finished jobs: how many, mean cost, minutes and tokens, and the money
        spent today and this month (local time of the server)."""
        t = time.localtime()
        day = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
        month = time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        with self._conn() as c:
            rows = c.execute(
                """SELECT engine, COALESCE(model, '') AS model, COUNT(*) AS n, AVG(cost_usd) AS cost,
                          AVG(NULLIF(minutes, -1.0)) AS minutes, AVG(tokens) AS tokens,
                          SUM(CASE WHEN updated >= ? THEN cost_usd ELSE 0 END) AS today,
                          SUM(CASE WHEN updated >= ? THEN cost_usd ELSE 0 END) AS month
                   FROM ai_jobs WHERE status = 'done' GROUP BY engine, COALESCE(model, '')""", (day, month)).fetchall()
        return [dict(r) for r in rows]

    def job_claim(self, worker: str, engines=("claude",)) -> Optional[dict]:
        """The oldest queued job for one of `engines`, now running for `worker`."""
        engines = [e for e in engines if e in ENGINES] or ["claude"]
        with self._lock, self._conn() as c:
            self._expire_stale(c)
            r = c.execute(f"SELECT * FROM ai_jobs WHERE status = 'queued' AND engine IN ({','.join('?' * len(engines))}) "
                          "ORDER BY id LIMIT 1", engines).fetchone()
            if r is None:
                return None
            c.execute("UPDATE ai_jobs SET status = 'running', worker = ?, message = 'começando', updated = ? WHERE id = ?",
                      (worker, time.time(), r["id"]))
            r = c.execute("SELECT * FROM ai_jobs WHERE id = ?", (r["id"],)).fetchone()
        return self._job(r)

    def job_update(self, jid: int, status: Optional[str] = None, message: Optional[str] = None,
                   result: Optional[dict] = None, applied: Optional[bool] = None,
                   cost_usd: Optional[float] = None, tokens: Optional[int] = None,
                   minutes: Optional[float] = None) -> Optional[dict]:
        if status is not None and status not in JOB_STATUSES:
            raise ValueError(f"status must be one of {JOB_STATUSES}")
        sets, args = ["updated = ?"], [time.time()]
        for col, val in (("status", status), ("message", message)):
            if val is not None:
                sets.append(f"{col} = ?")
                args.append(val)
        if result is not None:
            sets.append("result = ?")
            args.append(json.dumps(result, ensure_ascii=False))
        if applied is not None:
            sets.append("applied = ?")
            args.append(int(applied))
        for col, val in (("cost_usd", cost_usd), ("tokens", tokens), ("minutes", minutes)):
            if val is not None:
                sets.append(f"{col} = ?")
                args.append(val)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE ai_jobs SET {', '.join(sets)} WHERE id = ?", (*args, jid))
        return self.job_get(jid)
