"""
AI Memory Backend — เก็บข้อมูลฝึก + ความจำให้ AI
รัน:  pip install fastapi uvicorn
      API_KEY=รหัสลับ uvicorn main:app --host 0.0.0.0 --port 8000
วาง persona.txt ไว้โฟลเดอร์เดียวกัน (บุคลิก Vex GPT จะถูกต่อเข้า /chat อัตโนมัติ)
เอกสาร API: http://localhost:8000/docs
"""
import hashlib
import json
import os
import sqlite3
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

DB_PATH = os.getenv("DB_PATH", "memory.db")
API_KEY = os.getenv("API_KEY", "change-me")

app = FastAPI(title="AI Memory Backend")


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init():
    con = db()
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
    # ฐานข้อมูลเก่าที่สร้างด้วย tokenizer แบบเดิม → สร้างดัชนีใหม่เป็น trigram (ข้อมูลไม่หาย)
    row = con.execute("SELECT sql FROM sqlite_master WHERE name='samples_fts'").fetchone()
    if row and "trigram" not in (row[0] or ""):
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


init()

if API_KEY == "change-me":
    print("⚠️  API_KEY ยังเป็นค่าเริ่มต้น 'change-me' — ตั้งรหัสด้วย API_KEY=... ก่อนเปิดให้คนอื่นเข้าถึง")


def auth(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "bad api key")


class Sample(BaseModel):
    prompt: Optional[str] = ""
    response: str
    tag: str = "general"
    source: str = "manual"


@app.post("/samples", dependencies=[Depends(auth)])
def add(s: Sample):
    h = hashlib.sha256(((s.prompt or "") + "\x00" + s.response).encode()).hexdigest()
    con = db()
    try:
        cur = con.execute(
            "INSERT INTO samples(hash,prompt,response,tag,source,created) VALUES(?,?,?,?,?,?)",
            (h, s.prompt, s.response, s.tag, s.source, time.time()),
        )
        con.commit()
        return {"id": cur.lastrowid, "duplicate": False}
    except sqlite3.IntegrityError:
        return {"duplicate": True}
    finally:
        con.close()


@app.post("/samples/bulk", dependencies=[Depends(auth)])
def bulk(items: list[Sample]):
    added = sum(0 if add(i).get("duplicate") else 1 for i in items)
    return {"added": added, "skipped": len(items) - added}


@app.get("/samples", dependencies=[Depends(auth)])
def list_samples(tag: Optional[str] = None, limit: int = 50, offset: int = 0):
    con = db()
    q, a = "SELECT * FROM samples", []
    if tag:
        q += " WHERE tag=?"
        a.append(tag)
    q += " ORDER BY id DESC LIMIT ? OFFSET ?"
    rows = con.execute(q, a + [limit, offset]).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.delete("/samples/{sid}", dependencies=[Depends(auth)])
def delete(sid: int):
    con = db()
    con.execute("DELETE FROM samples WHERE id=?", (sid,))
    con.commit()
    con.close()
    return {"deleted": sid}


def fts_query(q: str, max_terms: int = 60) -> str:
    """แตกข้อความเป็นชิ้นละ 3 ตัวอักษร (trigram) ใช้ค้นภาษาไทยที่ไม่มีเว้นวรรคได้"""
    terms = []
    for chunk in q.replace('"', " ").split():
        if len(chunk) < 3:
            continue
        for i in range(len(chunk) - 2):
            t = chunk[i : i + 3]
            if t not in terms:
                terms.append(t)
    return " OR ".join(f'"{t}"' for t in terms[:max_terms])


@app.get("/recall", dependencies=[Depends(auth)])
def recall(q: str, k: int = 5):
    """ให้ AI ค้นความจำที่เกี่ยวข้อง (เอาไปต่อใน prompt ก่อนตอบ)"""
    con = db()
    words = fts_query(q)
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


@app.get("/stats", dependencies=[Depends(auth)])
def stats():
    con = db()
    total = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    tags = con.execute("SELECT tag, COUNT(*) c FROM samples GROUP BY tag").fetchall()
    con.close()
    return {"total": total, "tags": {r["tag"]: r["c"] for r in tags}}


@app.get("/export/jsonl", dependencies=[Depends(auth)])
def export_jsonl(tag: Optional[str] = None, messages: bool = False):
    """สำหรับ fine-tune: {"prompt":..., "response":...} ต่อบรรทัด
    ใส่ ?messages=true เพื่อได้รูปแบบ {"messages":[...]} ที่ใช้กับ finetune.py (ข้ามรายการที่ไม่มี prompt)"""
    con = db()
    q, a = "SELECT prompt,response FROM samples", []
    if tag:
        q += " WHERE tag=?"
        a.append(tag)
    rows = con.execute(q, a).fetchall()
    con.close()

    def gen():
        for r in rows:
            if messages:
                if not (r["prompt"] or "").strip():
                    continue
                d = {"messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["response"]},
                ]}
            else:
                d = dict(r)
            yield json.dumps(d, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/export/nanogpt", dependencies=[Depends(auth)], response_class=PlainTextResponse)
def export_nanogpt(tag: Optional[str] = None):
    """ข้อความดิบ → เซฟเป็น input.txt แล้วใช้กับ data/shakespeare_char/prepare.py ของ nanoGPT"""
    con = db()
    q, a = "SELECT prompt,response FROM samples", []
    if tag:
        q += " WHERE tag=?"
        a.append(tag)
    rows = con.execute(q, a).fetchall()
    con.close()
    parts = []
    for r in rows:
        p = (r["prompt"] or "").strip()
        parts.append((f"User: {p}\nAI: " if p else "") + r["response"].strip())
    return "\n\n".join(parts)


# ───────────── จำอัตโนมัติ: ใช้ /chat ที่เดียว ไม่ต้องเรียกคำสั่งอื่น ─────────────
import urllib.request

LLM_URL = os.getenv("LLM_URL", "")      # เช่น http://localhost:9000/v1/chat/completions (OpenAI-compatible)
LLM_KEY = os.getenv("LLM_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "my-model")
# ค่าเริ่มต้น "0": ไม่เก็บคำตอบของ AI เองเป็นความจำ (กันความผิดสะสม)
# เก็บเฉพาะที่ผู้ใช้สอนตรง ๆ ("จำไว้ว่า ...") ถ้าอยากเก็บแชททั้งหมดตั้ง AUTO_SAVE=1
AUTO_SAVE = os.getenv("AUTO_SAVE", "0") == "1"

# ── 2) บุคลิก Vex GPT: อ่านจาก persona.txt (หรือกำหนด PERSONA_PATH)
_here = os.path.dirname(os.path.abspath(__file__))
PERSONA_PATH = os.getenv("PERSONA_PATH", os.path.join(_here, "persona.txt"))
try:
    with open(PERSONA_PATH, encoding="utf-8") as _f:
        PERSONA = _f.read().strip()
except OSError:
    PERSONA = ""

# ประวัติแชทต่อ session เก็บในหน่วยความจำ (ไม่ปนกับความจำถาวรใน DB)
HISTORY: dict = {}
HISTORY_TURNS = 6


class ChatIn(BaseModel):
    message: str
    session: str = "default"


def call_llm(messages: list) -> str:
    if not LLM_URL:  # ยังไม่ได้ต่อโมเดล → ตอบ stub ไว้ก่อน
        return "(ยังไม่ได้ตั้ง LLM_URL) ความจำที่ดึงมา:\n" + messages[0]["content"]
    req = urllib.request.Request(
        LLM_URL,
        data=json.dumps({"model": LLM_MODEL, "messages": messages}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


@app.post("/chat", dependencies=[Depends(auth)])
def chat(c: ChatIn):
    # 1) ดึงความจำที่เกี่ยวข้องอัตโนมัติ
    mem = recall(c.message, k=5)
    mem_text = "\n".join(
        f"- {m['prompt'] or ''} → {m['response']}" if m["prompt"] else f"- {m['response']}"
        for m in mem
    )
    system = (PERSONA + "\n\n" if PERSONA else "") + (
        "ความรู้ที่เคยถูกสอนไว้ (ใช้ตอบถ้าเกี่ยวข้อง ถ้าไม่เกี่ยวให้ไม่ต้องพูดถึง):\n"
        + (mem_text or "(ไม่มี)")
    )

    # 2) ประวัติแชทล่าสุดของ session (จากหน่วยความจำของเซิร์ฟเวอร์)
    hist = HISTORY.setdefault(c.session, [])
    msgs = [{"role": "system", "content": system}]
    for u, a in hist[-HISTORY_TURNS:]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": c.message})

    reply = call_llm(msgs)
    hist.append((c.message, reply))
    del hist[:-HISTORY_TURNS]

    # 3) จำอัตโนมัติ
    if AUTO_SAVE:
        add(Sample(prompt=c.message, response=reply, tag="chat", source=f"chat:{c.session}"))
    # ถ้าผู้ใช้สอนตรง ๆ เช่น "จำไว้ว่า ..." → เก็บเป็นความรู้ถาวรด้วย
    for key in ("จำไว้ว่า", "จำไว้", "remember that"):
        if c.message.strip().startswith(key):
            fact = c.message.strip()[len(key):].strip(" :,")
            if fact:
                add(Sample(prompt="", response=fact, tag="taught", source="user-taught"))
            break
    return {"reply": reply, "used_memory": len(mem)}
