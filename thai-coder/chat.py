# chat.py — คุยกับโมเดลที่เทรนเสร็จแล้ว (รันใน Colab หลัง finetune.py)
#   ถามครั้งเดียว:    !python chat.py ข้อความที่ต้องการถาม
#   โหมดคุยต่อเนื่อง:  !python chat.py   (พิมพ์ exit เพื่อออก)
#   ให้ค้นเว็บก่อนตอบ: ขึ้นต้นข้อความด้วย "ค้น " เช่น  ค้น Roblox TweenService วิธีใช้
#   ต้องมีไฟล์ search.py อยู่โฟลเดอร์เดียวกัน และ !pip install ddgs

import os
import sys
import warnings
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "0"
warnings.filterwarnings("ignore")

# ดู log เต็ม ๆ เวลามีปัญหา:  !CHAT_DEBUG=1 python chat.py สวัสดี
DEBUG = os.environ.get("CHAT_DEBUG") == "1"

ADAPTER = "lora-th-coder"  # โฟลเดอร์ที่ finetune.py สร้างไว้

# บุคลิกของ AI อ่านจากไฟล์ persona.txt (แก้ไฟล์นั้นได้เลย)
_here = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(_here, "persona.txt"), encoding="utf-8") as _f:
        PERSONA = _f.read().strip()
except OSError:
    PERSONA = "คุณคือผู้ช่วยเขียนโค้ดภาษาไทย พูดสั้น ตรงประเด็น ลงท้ายด้วย 'ครับ'"


def quiet():
    if DEBUG:
        return nullcontext()
    stack = ExitStack()
    null = stack.enter_context(open(os.devnull, "w"))
    stack.enter_context(redirect_stdout(null))
    stack.enter_context(redirect_stderr(null))
    return stack


def show():
    if not DEBUG:
        print("AI is thinking...", end="", flush=True)


def hide():
    if not DEBUG:
        print("\r\033[K", end="", flush=True)


show()

# ซ่อนข้อความของ Unsloth/torch ตอนโหลดโมเดล (ถ้าอยากดู log ให้ลบบรรทัด with ออก)
with quiet():
    from unsloth import FastLanguageModel
    from search import build_prompt

    model, tok = FastLanguageModel.from_pretrained(ADAPTER, load_in_4bit=True)
    FastLanguageModel.for_inference(model)


def prep(text):
    if text.startswith("ค้น "):
        return build_prompt(text[4:].strip())
    return text


def ask(history):
    messages = [{"role": "system", "content": PERSONA}] + history
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    with quiet():
        out = model.generate(
            ids, max_new_tokens=512, max_length=None, temperature=0.3, do_sample=True
        )
    reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return reply or "(โมเดลไม่ตอบ ลองถามใหม่ หรือเพิ่มข้อมูลเทรน)"


if len(sys.argv) > 1:  # ถามครั้งเดียวแล้วจบ
    text = " ".join(sys.argv[1:])
    reply = ask([{"role": "user", "content": prep(text)}])
    hide()
    print(reply)
else:  # โหมดคุยต่อเนื่อง
    hide()
    history = []
    while True:
        text = input("คุณ: ").strip()
        if text.lower() in ("exit", "quit", "ออก"):
            break
        if not text:
            continue
        history.append({"role": "user", "content": prep(text)})
        show()
        reply = ask(history)
        hide()
        print("AI:", reply)
        history.append({"role": "assistant", "content": reply})
