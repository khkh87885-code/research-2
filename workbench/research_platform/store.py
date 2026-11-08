import json
import sqlite3
from pathlib import Path
from .common import now


class Store:
    def __init__(self, directory):
        self.path = Path(directory) / "tasks.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS tasks (image_key TEXT, variant TEXT, status TEXT, attempts INTEGER DEFAULT 0, result TEXT, PRIMARY KEY(image_key,variant))")
            db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, time TEXT, image_key TEXT, variant TEXT, status TEXT, detail TEXT)")

    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def schedule(self, records, variants):
        with self.connect() as db:
            for row in records:
                for variant in variants:
                    db.execute("INSERT OR IGNORE INTO tasks(image_key,variant,status) VALUES(?,?,'pending')", (row["dataset"] + ":" + row["image_id"], variant))

    def state(self, key, variant):
        with self.connect() as db:
            return dict(db.execute("SELECT * FROM tasks WHERE image_key=? AND variant=?", (key, variant)).fetchone())

    def update(self, key, variant, status, result=None):
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT status FROM tasks WHERE image_key=? AND variant=?", (key, variant)).fetchone()
            if current is None:
                raise KeyError("Unknown task")
            if current["status"] == "completed":
                raise ValueError("Completed task records are immutable; create a new experiment version")
            db.execute("UPDATE tasks SET status=?,attempts=attempts+?,result=COALESCE(?,result) WHERE image_key=? AND variant=?", (status, int(status == "running"), payload, key, variant))
            db.execute("INSERT INTO events(time,image_key,variant,status,detail) VALUES(?,?,?,?,?)", (now(), key, variant, status, payload))

    def results(self):
        with self.connect() as db:
            records = db.execute("SELECT * FROM tasks ORDER BY image_key,variant").fetchall()
        return [dict(json.loads(r["result"]) if r["result"] else {}, image_key=r["image_key"], variant=r["variant"], status=r["status"], attempts=r["attempts"]) for r in records]

    def summary(self):
        with self.connect() as db:
            counts = {r["status"]: r["n"] for r in db.execute("SELECT status,COUNT(*) n FROM tasks GROUP BY status")}
        total = sum(counts.values())
        done = counts.get("completed", 0) + counts.get("failed", 0)
        return {"counts": counts, "total": total, "finished": done, "progress": done / total if total else 0,
                "status": "completed_with_failures" if done == total and counts.get("failed") else "completed" if done == total else "running"}
