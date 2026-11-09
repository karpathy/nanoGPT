# chat.py — คุยกับโมเดลที่เทรนเสร็จแล้ว (รันใน Colab หลัง finetune.py)
#   ถามครั้งเดียว:  !python chat.py ข้อความที่ต้องการถาม
#   โหมดคุยต่อเนื่อง: !python chat.py   (พิมพ์ exit เพื่อออก)

import sys
from unsloth import FastLanguageModel

ADAPTER = "lora-th-coder"  # โฟลเดอร์ที่ finetune.py สร้างไว้

model, tok = FastLanguageModel.from_pretrained(ADAPTER, load_in_4bit=True)
FastLanguageModel.for_inference(model)


def ask(history):
    ids = tok.apply_chat_template(
        history, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(ids, max_new_tokens=512, temperature=0.7, do_sample=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


if len(sys.argv) > 1:  # ถามครั้งเดียวแล้วจบ
    print(ask([{"role": "user", "content": " ".join(sys.argv[1:])}]))
else:  # โหมดคุยต่อเนื่อง
    history = []
    while True:
        text = input("คุณ: ").strip()
        if text.lower() in ("exit", "quit", "ออก"):
            break
        if not text:
            continue
        history.append({"role": "user", "content": text})
        reply = ask(history)
        print("AI:", reply)
        history.append({"role": "assistant", "content": reply})
