import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager


class Store:
    def __init__(self, path, seed):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, title TEXT, text TEXT, source TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, payload TEXT)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def documents(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return self.seed + [dict(row) for row in db.execute('SELECT * FROM documents ORDER BY rowid')]

    def add_document(self, title, text, source):
        doc = dict(id=uuid.uuid4().hex, title=title, text=text, source=source)
        with self.connect() as db:
            db.execute('INSERT INTO documents VALUES (:id,:title,:text,:source)', doc)
        return doc

    def save_run(self, result):
        result['id'] = uuid.uuid4().hex
        result['created_at'] = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO runs VALUES (?,?,?)', (result['id'], result['created_at'], json.dumps(result)))
        return result

    def history(self, limit=20):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM runs ORDER BY rowid DESC LIMIT ?', (limit,))]

