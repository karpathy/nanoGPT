# serve.py — เปิดโมเดล Vex GPT เป็น API แบบ OpenAI-compatible
#   POST /v1/chat/completions   (main.py เรียกตัวนี้ผ่าน LLM_URL)
#   รัน:  uvicorn serve:app --host 127.0.0.1 --port 9000
#   ต้องมี GPU และโฟลเดอร์ lora-th-coder (ได้จาก finetune.py)

import os
import sys
import threading
import time
import warnings
from typing import List

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch

if not torch.cuda.is_available():
    sys.exit("ไม่พบ GPU ไปที่ Runtime > Change runtime type > T4 GPU แล้วรันใหม่")

_here = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.getenv("ADAPTER", os.path.join(_here, "lora-th-coder"))
MODEL_NAME = os.getenv("LLM_MODEL", "vex-gpt")

if not os.path.isdir(ADAPTER):
    sys.exit("ไม่พบโฟลเดอร์ lora-th-coder ต้องรัน  python finetune.py  ก่อน")

from unsloth import FastLanguageModel  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

model, tok = FastLanguageModel.from_pretrained(ADAPTER, load_in_4bit=True)
FastLanguageModel.for_inference(model)

lock = threading.Lock()  # GPU ตัวเดียว ตอบทีละคำขอ
app = FastAPI(title="Vex GPT server")


class Msg(BaseModel):
    role: str
    content: str


class Req(BaseModel):
    model: str = MODEL_NAME
    messages: List[Msg]
    max_tokens: int = 512
    temperature: float = 0.3


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/v1/chat/completions")
def completions(r: Req):
    msgs = [{"role": m.role, "content": m.content} for m in r.messages]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    with lock:
        out = model.generate(
            ids,
            max_new_tokens=r.max_tokens,
            max_length=None,
            temperature=max(r.temperature, 0.01),
            do_sample=True,
            repetition_penalty=1.15,
        )
    reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    reply = reply.replace("。", " ").strip()
    n_in, n_out = int(ids.shape[1]), int(out.shape[1] - ids.shape[1])
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }
