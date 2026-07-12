import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "app.db"

DEFAULT_CATEGORIES = ["accord", "refus", "demande de pièces", "relance", "autre"]

STATUSES = ["nouveau", "brouillon", "traité"]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                filename TEXT,
                raw_text TEXT NOT NULL,
                category TEXT,
                urgency TEXT,
                dossier_ref TEXT,
                summary TEXT,
                status TEXT NOT NULL DEFAULT 'nouveau',
                draft TEXT
            )"""
        )
        conn.execute("CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY)")
        if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO categories (name) VALUES (?)",
                [(c,) for c in DEFAULT_CATEGORIES],
            )


def insert_response(
    filename: str | None,
    raw_text: str,
    category: str,
    urgency: str,
    dossier_ref: str,
    summary: str,
) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO responses (filename, raw_text, category, urgency, dossier_ref, summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (filename, raw_text, category, urgency, dossier_ref, summary),
        )
        new_id = cur.lastrowid
    # hors du bloc with : la transaction est commitée, la ligne est visible
    return get_response(new_id)


def get_response(response_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM responses WHERE id = ?", (response_id,)
        ).fetchone()
        return dict(row) if row else None


def list_responses(category: str | None = None, status: str | None = None) -> list[dict]:
    query = "SELECT * FROM responses WHERE 1=1"
    params: list = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def update_response(response_id: int, **fields) -> dict | None:
    allowed = {"draft", "status", "category", "dossier_ref"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if updates:
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with get_conn() as conn:
            conn.execute(
                f"UPDATE responses SET {assignments} WHERE id = ?",
                (*updates.values(), response_id),
            )
    return get_response(response_id)


def get_categories() -> list[str]:
    with get_conn() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM categories ORDER BY name")]


def set_categories(names: list[str]):
    cleaned = [n.strip() for n in names if n.strip()]
    with get_conn() as conn:
        conn.execute("DELETE FROM categories")
        conn.executemany("INSERT INTO categories (name) VALUES (?)", [(n,) for n in cleaned])


def stats() -> dict:
    with get_conn() as conn:
        by_category = {
            r["category"] or "non classé": r["n"]
            for r in conn.execute(
                "SELECT category, COUNT(*) AS n FROM responses GROUP BY category"
            )
        }
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM responses GROUP BY status"
            )
        }
        total = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    return {"total": total, "by_category": by_category, "by_status": by_status}
