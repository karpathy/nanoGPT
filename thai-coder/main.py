"""
AI Memory Backend — เก็บข้อมูลฝึก + ความจำให้ AI (รองรับ Vercel)

Vercel:  ตั้ง Root Directory = thai-coder/backend
         ต่อ Upstash Redis (Storage/Marketplace) แล้วตั้ง Environment Variables:
           API_KEY=รหัสลับ   LLM_URL=https://ลิงก์โมเดลของคุณ/v1/chat/completions
Local :  pip install fastapi uvicorn
         API_KEY=รหัสลับ uvicorn main:app --host 0.0.0.0 --port 8000   (ใช้ SQLite memory.db)
บุคลิก Vex GPT อ่านจาก persona.txt ในโฟลเดอร์เดียวกัน (หรือ env PERSONA)
"""
import json
import os
import urllib.error
import urllib.request
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from store import make_store

API_KEY = os.getenv("API_KEY", "change-me")
LLM_URL = os.getenv("LLM_URL", "")  # OpenAI-compatible เช่น https://xxx/v1/chat/completions
LLM_KEY = os.getenv("LLM_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "my-model")
# ค่าเริ่มต้น "0": ไม่เก็บคำตอบของ AI เองเป็นความจำ (กันความผิดสะสม)
AUTO_SAVE = os.getenv("AUTO_SAVE", "0") == "1"
HISTORY_TURNS = 6

_here = os.path.dirname(os.path.abspath(__file__))
PERSONA = os.getenv("PERSONA", "").strip()
if not PERSONA:
    try:
        with open(os.getenv("PERSONA_PATH", os.path.join(_here, "persona.txt")), encoding="utf-8") as _f:
            PERSONA = _f.read().strip()
    except OSError:
        PERSONA = ""

STORE = make_store()

if API_KEY == "change-me":
    print("⚠️  API_KEY ยังเป็นค่าเริ่มต้น 'change-me' — ตั้งรหัสก่อนเปิดให้คนอื่นเข้าถึง")

app = FastAPI(title="AI Memory Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def auth(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "bad api key")


class Sample(BaseModel):
    prompt: Optional[str] = ""
    response: str
    tag: str = "general"
    source: str = "manual"


def _add(s: Sample):
    return STORE.add(s.prompt or "", s.response, s.tag, s.source)


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "store": STORE.kind}


@app.post("/samples", dependencies=[Depends(auth)])
def add(s: Sample):
    return _add(s)


@app.post("/samples/bulk", dependencies=[Depends(auth)])
def bulk(items: list[Sample]):
    added = sum(0 if _add(i).get("duplicate") else 1 for i in items)
    return {"added": added, "skipped": len(items) - added}


@app.get("/samples", dependencies=[Depends(auth)])
def list_samples(tag: Optional[str] = None, limit: int = 50, offset: int = 0):
    return STORE.list(tag, limit, offset)


@app.delete("/samples/{sid}", dependencies=[Depends(auth)])
def delete(sid: int):
    STORE.delete(sid)
    return {"deleted": sid}


@app.get("/recall", dependencies=[Depends(auth)])
def recall(q: str, k: int = 5):
    """ให้ AI ค้นความจำที่เกี่ยวข้อง (เอาไปต่อใน prompt ก่อนตอบ)"""
    return STORE.recall(q, k)


@app.get("/stats", dependencies=[Depends(auth)])
def stats():
    return STORE.stats()


@app.get("/export/jsonl", dependencies=[Depends(auth)])
def export_jsonl(tag: Optional[str] = None, messages: bool = False):
    """สำหรับ fine-tune: {"prompt":..., "response":...} ต่อบรรทัด
    ใส่ ?messages=true เพื่อได้รูปแบบ {"messages":[...]} ที่ใช้กับ finetune.py (ข้ามรายการที่ไม่มี prompt)"""
    lines = []
    for r in STORE.rows(tag):
        if messages:
            if not (r["prompt"] or "").strip():
                continue
            d = {"messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ]}
        else:
            d = {"prompt": r["prompt"], "response": r["response"]}
        lines.append(json.dumps(d, ensure_ascii=False) + "\n")
    return Response("".join(lines), media_type="application/x-ndjson")


@app.get("/export/nanogpt", dependencies=[Depends(auth)], response_class=PlainTextResponse)
def export_nanogpt(tag: Optional[str] = None):
    """ข้อความดิบ → เซฟเป็น input.txt แล้วใช้กับ data/shakespeare_char/prepare.py ของ nanoGPT"""
    parts = []
    for r in STORE.rows(tag):
        p = (r["prompt"] or "").strip()
        parts.append((f"User: {p}\nAI: " if p else "") + r["response"].strip())
    return "\n\n".join(parts)


# ───────────── จำอัตโนมัติ: ใช้ /chat ที่เดียว ไม่ต้องเรียกคำสั่งอื่น ─────────────
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
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        raise HTTPException(502, f"เรียกโมเดลไม่สำเร็จ: {e}")


@app.post("/chat", dependencies=[Depends(auth)])
def chat(c: ChatIn):
    # 1) ดึงความจำที่เกี่ยวข้องอัตโนมัติ
    mem = STORE.recall(c.message, k=5)
    mem_text = "\n".join(
        f"- {m['prompt'] or ''} → {m['response']}" if m["prompt"] else f"- {m['response']}"
        for m in mem
    )
    system = (PERSONA + "\n\n" if PERSONA else "") + (
        "ความรู้ที่เคยถูกสอนไว้ (ใช้ตอบถ้าเกี่ยวข้อง ถ้าไม่เกี่ยวให้ไม่ต้องพูดถึง):\n"
        + (mem_text or "(ไม่มี)")
    )

    # 2) ประวัติแชทล่าสุดของ session
    hist = STORE.get_hist(c.session)
    msgs = [{"role": "system", "content": system}]
    for u, a in hist[-HISTORY_TURNS:]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": c.message})

    reply = call_llm(msgs)
    STORE.set_hist(c.session, (hist + [[c.message, reply]])[-HISTORY_TURNS:])

    # 3) จำอัตโนมัติ
    if AUTO_SAVE:
        STORE.add(c.message, reply, "chat", f"chat:{c.session}")
    # ถ้าผู้ใช้สอนตรง ๆ เช่น "จำไว้ว่า ..." → เก็บเป็นความรู้ถาวรด้วย
    for key in ("จำไว้ว่า", "จำไว้", "remember that"):
        if c.message.strip().startswith(key):
            fact = c.message.strip()[len(key):].strip(" :,")
            if fact:
                STORE.add("", fact, "taught", "user-taught")
            break
    return {"reply": reply, "used_memory": len(mem)}
