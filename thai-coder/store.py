"""ที่เก็บความจำ 2 แบบ (เลือกอัตโนมัติจาก environment)

- RedisStore  : ใช้ Upstash Redis (REST) → เหมาะกับ Vercel ที่เขียนไฟล์ถาวรไม่ได้
                ตั้ง UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN
                (หรือ KV_REST_API_URL + KV_REST_API_TOKEN ที่ Vercel ใส่ให้ตอนต่อ Upstash)
- SqliteStore : ใช้ไฟล์ memory.db → ใช้ตอนรันเองบน Colab/เครื่องตัวเอง/โฮสต์ที่มีดิสก์ถาวร
"""
import hashlib
import json
import os
import sqlite3
import time
import urllib.request


def sample_hash(prompt, response):
    return hashlib.sha256(((prompt or "") + "\x00" + response).encode()).hexdigest()


def trigram_list(text, max_terms=60):
    """แตกข้อความเป็นชิ้นละ 3 ตัวอักษร ใช้ค้นภาษาไทยที่ไม่มีเว้นวรรคได้"""
    seen, out = set(), []
    for chunk in (text or "").replace('"', " ").split():
        for i in range(len(chunk) - 2):
            t = chunk[i : i + 3]
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out[:max_terms]


# ───────────────────────── SQLite ─────────────────────────
class SqliteStore:
    kind = "sqlite"

    def __init__(self, path):
        self.path = path
        self._hist = {}  # ประวัติแชทต่อ session (ในหน่วยความจำ)
        self._init()

    def _db(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init(self):
        con = self._db()
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE,
                prompt TEXT,
                response TEXT NOT NULL,
                tag TEXT DEFAULT 'general',
                source TEXT DEFAULT 'manual',
                created REAL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS samples_fts
                USING fts5(prompt, response, content='samples', content_rowid='id',
                           tokenize='trigram');
            CREATE TRIGGER IF NOT EXISTS s_ai AFTER INSERT ON samples BEGIN
                INSERT INTO samples_fts(rowid, prompt, response)
                VALUES (new.id, new.prompt, new.response);
            END;
            CREATE TRIGGER IF NOT EXISTS s_ad AFTER DELETE ON samples BEGIN
                INSERT INTO samples_fts(samples_fts, rowid, prompt, response)
                VALUES ('delete', old.id, old.prompt, old.response);
            END;
            """
        )
        con.commit()
        row = con.execute("SELECT sql FROM sqlite_master WHERE name='samples_fts'").fetchone()
        if row and "trigram" not in (row[0] or ""):  # ฐานข้อมูลเก่า → สร้างดัชนีใหม่
            con.executescript(
                """
                DROP TABLE samples_fts;
                CREATE VIRTUAL TABLE samples_fts
                    USING fts5(prompt, response, content='samples', content_rowid='id',
                               tokenize='trigram');
                INSERT INTO samples_fts(samples_fts) VALUES('rebuild');
                """
            )
            con.commit()
        con.close()

    def add(self, prompt, response, tag="general", source="manual"):
        con = self._db()
        try:
            cur = con.execute(
                "INSERT INTO samples(hash,prompt,response,tag,source,created) VALUES(?,?,?,?,?,?)",
                (sample_hash(prompt, response), prompt, response, tag, source, time.time()),
            )
            con.commit()
            return {"id": cur.lastrowid, "duplicate": False}
        except sqlite3.IntegrityError:
            return {"duplicate": True}
        finally:
            con.close()

    def list(self, tag=None, limit=50, offset=0):
        con = self._db()
        q, a = "SELECT * FROM samples", []
        if tag:
            q += " WHERE tag=?"
            a.append(tag)
        rows = con.execute(q + " ORDER BY id DESC LIMIT ? OFFSET ?", a + [limit, offset]).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def rows(self, tag=None):
        con = self._db()
        q, a = "SELECT * FROM samples", []
        if tag:
            q += " WHERE tag=?"
            a.append(tag)
        rows = con.execute(q + " ORDER BY id", a).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def delete(self, sid):
        con = self._db()
        con.execute("DELETE FROM samples WHERE id=?", (sid,))
        con.commit()
        con.close()

    def recall(self, q, k=5):
        con = self._db()
        terms = trigram_list(q)
        words = " OR ".join(f'"{t}"' for t in terms)
        rows = []
        if words:
            try:
                rows = con.execute(
                    "SELECT s.* FROM samples_fts f JOIN samples s ON s.id=f.rowid "
                    "WHERE samples_fts MATCH ? ORDER BY rank LIMIT ?",
                    (words, k),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:  # ข้อความสั้นกว่า 3 ตัวอักษร → fallback LIKE
            like = f"%{q}%"
            rows = con.execute(
                "SELECT * FROM samples WHERE prompt LIKE ? OR response LIKE ? LIMIT ?",
                (like, like, k),
            ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def stats(self):
        con = self._db()
        total = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        tags = con.execute("SELECT tag, COUNT(*) c FROM samples GROUP BY tag").fetchall()
        con.close()
        return {"total": total, "tags": {r["tag"]: r["c"] for r in tags}}

    def get_hist(self, session):
        return list(self._hist.get(session, []))

    def set_hist(self, session, hist):
        self._hist[session] = hist


# ───────────────────────── Upstash Redis (REST) ─────────────────────────
class RedisStore:
    kind = "redis"
    SAMPLES = "vex:samples"  # hash: id -> json
    HASHES = "vex:hashes"  # set ไว้กันข้อมูลซ้ำ
    SEQ = "vex:seq"

    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.token = token

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)

    def _cmd(self, *args):
        res = self._post("", [str(a) for a in args])
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(res["error"])
        return res["result"]

    def _all(self):
        vals = self._cmd("HVALS", self.SAMPLES) or []
        return [json.loads(v) for v in vals]

    def add(self, prompt, response, tag="general", source="manual"):
        h = sample_hash(prompt, response)
        if int(self._cmd("SADD", self.HASHES, h)) == 0:
            return {"duplicate": True}
        sid = int(self._cmd("INCR", self.SEQ))
        row = {
            "id": sid, "hash": h, "prompt": prompt or "", "response": response,
            "tag": tag, "source": source, "created": time.time(),
        }
        self._cmd("HSET", self.SAMPLES, sid, json.dumps(row, ensure_ascii=False))
        return {"id": sid, "duplicate": False}

    def list(self, tag=None, limit=50, offset=0):
        rows = [r for r in self._all() if not tag or r["tag"] == tag]
        rows.sort(key=lambda r: r["id"], reverse=True)
        return rows[offset : offset + limit]

    def rows(self, tag=None):
        rows = [r for r in self._all() if not tag or r["tag"] == tag]
        rows.sort(key=lambda r: r["id"])
        return rows

    def delete(self, sid):
        raw = self._cmd("HGET", self.SAMPLES, sid)
        if raw:
            row = json.loads(raw)
            self._cmd("HDEL", self.SAMPLES, sid)
            self._cmd("SREM", self.HASHES, row["hash"])

    def recall(self, q, k=5):
        terms = set(trigram_list(q))
        rows = self._all()
        if not terms:
            return []
        scored = []
        for r in rows:
            text = (r["prompt"] or "") + " " + r["response"]
            hit = sum(1 for t in terms if t in text)
            if hit >= min(2, len(terms)):  # ต้องตรงอย่างน้อย 2 ชิ้น กันเจอมั่ว
                scored.append((hit, r["id"], r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored:
            return [r for _, _, r in scored[:k]]
        like = (q or "").lower()  # ข้อความสั้นมาก → ค้นแบบตรงตัว
        return [r for r in rows if like and like in ((r["prompt"] or "") + r["response"]).lower()][:k]

    def stats(self):
        rows = self._all()
        tags = {}
        for r in rows:
            tags[r["tag"]] = tags.get(r["tag"], 0) + 1
        return {"total": len(rows), "tags": tags}

    def get_hist(self, session):
        raw = self._cmd("GET", f"vex:hist:{session}")
        return json.loads(raw) if raw else []

    def set_hist(self, session, hist):
        self._cmd("SET", f"vex:hist:{session}", json.dumps(hist, ensure_ascii=False), "EX", 86400)


def make_store():
    url = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("KV_REST_API_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("KV_REST_API_TOKEN")
    if url and token:
        return RedisStore(url, token)
    path = os.getenv("DB_PATH")
    if not path:
        path = "/tmp/memory.db" if os.getenv("VERCEL") else "memory.db"
    if os.getenv("VERCEL"):
        print("⚠️  Vercel: ยังไม่ได้ต่อ Upstash Redis — ข้อมูลใน /tmp จะหายเมื่อฟังก์ชันรีสตาร์ท")
    return SqliteStore(path)
