"""Persistência em SQLite: vagas vistas, notas, candidaturas e respostas reutilizáveis."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    location    TEXT,
    url         TEXT NOT NULL,
    posted      TEXT,
    score       REAL NOT NULL DEFAULT 0,
    reasons     TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',
    note        TEXT,
    seen_at     TEXT NOT NULL,
    applied_at  TEXT
);

CREATE TABLE IF NOT EXISTS answers (
    question TEXT PRIMARY KEY,
    answer   TEXT NOT NULL,
    source   TEXT NOT NULL,
    used_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_applied_at ON jobs(applied_at);
"""

# queued  -> passou no filtro, aguardando candidatura
# skipped -> descartado pelo filtro/nota
# filled  -> formulário preenchido, aguardando revisão humana (dry-run)
# applied -> enviado
# failed  -> erro no preenchimento ou formulário não suportado
Status = str


@dataclass
class Job:
    job_id: str
    title: str
    company: str
    url: str
    location: str = ""
    posted: str = ""
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    status: Status = "queued"
    note: str = ""
    applied_at: str | None = None


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        with closing(self.conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- jobs

    def has_seen(self, job_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row is not None

    def upsert_job(self, job: Job) -> None:
        """Insere a vaga. Se já existe, preserva status e histórico."""
        self.conn.execute(
            """
            INSERT INTO jobs (job_id, title, company, location, url, posted,
                              score, reasons, status, note, seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                score = excluded.score,
                reasons = excluded.reasons
            """,
            (
                job.job_id,
                job.title,
                job.company,
                job.location,
                job.url,
                job.posted,
                job.score,
                json.dumps(job.reasons, ensure_ascii=False),
                job.status,
                job.note,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def set_status(self, job_id: str, status: Status, note: str = "") -> None:
        applied_at = (
            datetime.now().isoformat(timespec="seconds") if status == "applied" else None
        )
        self.conn.execute(
            """
            UPDATE jobs
               SET status = ?,
                   note = ?,
                   applied_at = COALESCE(?, applied_at)
             WHERE job_id = ?
            """,
            (status, note, applied_at, job_id),
        )
        self.conn.commit()

    def queued(self, limit: int, min_score: float = 0.0) -> list[Job]:
        rows = self.conn.execute(
            """
            SELECT * FROM jobs
             WHERE status = 'queued' AND score >= ?
             ORDER BY score DESC
             LIMIT ?
            """,
            (min_score, limit),
        ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def by_status(self, status: Status | None = None, limit: int = 100) -> list[Job]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY score DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY seen_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def applied_today(self) -> int:
        today = date.today().isoformat()
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = 'applied' AND applied_at LIKE ?",
            (f"{today}%",),
        ).fetchone()
        return int(row["n"])

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"],
            title=row["title"],
            company=row["company"],
            url=row["url"],
            location=row["location"] or "",
            posted=row["posted"] or "",
            score=row["score"],
            reasons=json.loads(row["reasons"] or "[]"),
            status=row["status"],
            note=row["note"] or "",
            applied_at=row["applied_at"],
        )

    # ------------------------------------------------------------- answers

    def remember_answer(self, question: str, answer: str, source: str) -> None:
        """Guarda a resposta para reaproveitar em formulários futuros."""
        self.conn.execute(
            """
            INSERT INTO answers (question, answer, source, used_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(question) DO UPDATE SET
                answer = excluded.answer,
                source = excluded.source,
                used_at = excluded.used_at
            """,
            (
                question.strip().lower(),
                answer,
                source,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def recall_answer(self, question: str) -> str | None:
        row = self.conn.execute(
            "SELECT answer FROM answers WHERE question = ?",
            (question.strip().lower(),),
        ).fetchone()
        return row["answer"] if row else None

    def all_answers(self) -> list[tuple[str, str, str]]:
        rows = self.conn.execute(
            "SELECT question, answer, source FROM answers ORDER BY used_at DESC"
        ).fetchall()
        return [(r["question"], r["answer"], r["source"]) for r in rows]
