# chat.py — คุยกับโมเดลที่เทรนเสร็จแล้ว (รันใน Colab หลัง finetune.py)
#   ถามครั้งเดียว:    !python chat.py ข้อความที่ต้องการถาม
#   โหมดคุยต่อเนื่อง:  !python chat.py   (พิมพ์ exit เพื่อออก)
#   ให้ค้นเว็บก่อนตอบ: ขึ้นต้นข้อความด้วย "ค้น " เช่น  ค้น Roblox TweenService วิธีใช้
#   ต้องมีไฟล์ search.py อยู่โฟลเดอร์เดียวกัน และ !pip install ddgs

import sys
from unsloth import FastLanguageModel
from search import build_prompt

ADAPTER = "lora-th-coder"  # โฟลเดอร์ที่ finetune.py สร้างไว้

model, tok = FastLanguageModel.from_pretrained(ADAPTER, load_in_4bit=True)
FastLanguageModel.for_inference(model)


def prep(text):
    if text.startswith("ค้น "):
        return build_prompt(text[4:].strip())
    return text


def ask(history):
    ids = tok.apply_chat_template(
        history, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(ids, max_new_tokens=512, temperature=0.7, do_sample=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


if len(sys.argv) > 1:  # ถามครั้งเดียวแล้วจบ
    text = " ".join(sys.argv[1:])
    print(ask([{"role": "user", "content": prep(text)}]))
else:  # โหมดคุยต่อเนื่อง
    history = []
    while True:
        text = input("คุณ: ").strip()
        if text.lower() in ("exit", "quit", "ออก"):
            break
        if not text:
            continue
        history.append({"role": "user", "content": prep(text)})
        reply = ask(history)
        print("AI:", reply)
        history.append({"role": "assistant", "content": reply})
