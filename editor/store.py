"""SQLite-backed store for the editor: one row per plan (a PDF page), files on disk.

Layout under ``EDITOR_DATA`` (default ``/data``):
  editor.sqlite            plans table
  pdfs/<id>.pdf            the uploaded PDF
  renders/<id>.png         base render of the plan region (long side 2000 px)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

STATUSES = ("pending", "corrected", "skipped")


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
        for p in (self.pdf_path(pid), self.render_path(pid)):
            if os.path.isfile(p):
                os.remove(p)
