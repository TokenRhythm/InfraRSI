"""Cross-task memory (memory.db, sqlite): tagged slow->fast pairs and failure reasons (FINAL_PLAN 3.2).

Entries carry hardware/model/data tags so retrieval matches by context. In held-out evaluation the database is
opened read-only and write_memory is a logged no-op (rsi.heldout).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS entries(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc TEXT NOT NULL,
  source TEXT NOT NULL,
  task_id TEXT,
  tags TEXT NOT NULL,
  slow TEXT,
  fast TEXT,
  outcome TEXT,
  failure_reason TEXT,
  lesson TEXT NOT NULL,
  digest TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
'''
SCHEMA_VERSION = 'infrarsi-memory-v1'
MAX_ENTRIES = 2000


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def create(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as con:
        con.executescript(SCHEMA_SQL)
        con.execute('INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)', ('schema', SCHEMA_VERSION))
    return p


def flat_tags(tags: dict | None) -> set[str]:
    out = set()
    for k, v in (tags or {}).items():
        for x in (v if isinstance(v, (list, tuple, set)) else [v]):
            if x is not None and x != '':
                out.add(f'{k}:{str(x).lower()}')
    return out


class MemoryDB:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path, self.read_only = Path(path), read_only
        if not self.path.exists():
            if read_only:
                raise FileNotFoundError(self.path)
            create(self.path)
        uri = f'file:{self.path.as_posix()}?mode=ro' if read_only else f'file:{self.path.as_posix()}'
        self.con = sqlite3.connect(uri, uri=True)
        self.con.row_factory = sqlite3.Row

    def close(self):
        self.con.close()

    def schema_ok(self) -> bool:
        try:
            row = self.con.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            cols = [r[1] for r in self.con.execute('PRAGMA table_info(entries)')]
        except sqlite3.DatabaseError:
            return False
        return row is not None and row[0] == SCHEMA_VERSION and cols[:3] == ['id', 'created_utc', 'source']

    def add(self, entry: dict, *, source: str, task_id: str | None) -> int | None:
        if self.read_only:
            return None
        body = {'tags': entry.get('tags') or {}, 'slow': entry.get('slow'), 'fast': entry.get('fast'),
                'outcome': entry.get('outcome'), 'failure_reason': entry.get('failure_reason'), 'lesson': entry['lesson']}
        dig = hashlib.sha256(_canon(body).encode('utf-8')).hexdigest()
        if self.count() >= MAX_ENTRIES:
            raise ValueError('memory.db entry limit reached')
        cur = self.con.execute(
            'INSERT OR IGNORE INTO entries(created_utc, source, task_id, tags, slow, fast, outcome, failure_reason, lesson, digest) '
            'VALUES (?,?,?,?,?,?,?,?,?,?)',
            (_dt.datetime.now(_dt.timezone.utc).isoformat(), source, task_id, _canon(body['tags']),
             None if body['slow'] is None else _canon(body['slow']), None if body['fast'] is None else _canon(body['fast']),
             body['outcome'], body['failure_reason'], body['lesson'], dig))
        self.con.commit()
        return cur.lastrowid if cur.rowcount else None

    def count(self) -> int:
        return self.con.execute('SELECT COUNT(*) FROM entries').fetchone()[0]

    def rows(self) -> list[dict]:
        out = []
        for r in self.con.execute('SELECT * FROM entries ORDER BY id'):
            d = dict(r)
            for k in ('tags', 'slow', 'fast'):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    def query(self, tags: dict, k: int = 3) -> list[dict]:
        """Top-k entries by tag overlap (Jaccard), newest first on ties."""
        want = flat_tags(tags)
        scored = []
        for d in self.rows():
            have = flat_tags(d['tags'])
            if not want or not have:
                continue
            j = len(want & have) / len(want | have)
            if j > 0:
                scored.append((j, d['id'], d))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [dict(d, score=round(j, 3)) for j, _, d in scored[:k]]

    def logical_digest(self) -> str:
        """Content digest independent of sqlite page layout."""
        rows = [{k: v for k, v in r.items() if k not in ('id', 'created_utc')} for r in self.rows()]
        return hashlib.sha256(_canon(rows).encode('utf-8')).hexdigest()


def logical_digest(path: str | Path) -> str:
    db = MemoryDB(path, read_only=True)
    try:
        return db.logical_digest()
    finally:
        db.close()
