# add_creator.py — เพิ่มตัวอย่าง "ใครคือผู้สร้าง/ผู้พัฒนา/ช่อง" ต่อท้าย data.jsonl
#   แก้ CREATOR และ CHANNEL ด้านล่าง แล้วรัน  !python add_creator.py
#   (หรือใส่ชื่อตอนรัน  !python add_creator.py "ชื่อผู้พัฒนา")
#   จากนั้นรัน  !python finetune.py  ใหม่

import json
import os
import sys

CREATOR = "เฟียสสุดหล่อ"  # ชื่อผู้พัฒนา/ผู้สร้าง
CHANNEL = "Fiatx2"  # ชื่อช่อง (ปล่อยว่างได้ถ้าไม่ต้องการ)

name = " ".join(sys.argv[1:]).strip() or CREATOR.strip()
channel = CHANNEL.strip()
if not name:
    sys.exit('ยังไม่ได้ใส่ชื่อ: แก้ CREATOR ในไฟล์ หรือรัน python add_creator.py "ชื่อ"')

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.jsonl")

who = f"{name} เจ้าของช่อง {channel}" if channel else name
who_en = f"{name}, owner of the {channel} channel" if channel else name

th_q = [
    "ใครเป็นผู้สร้างคุณ", "ใครสร้างคุณ", "ใครเป็นผู้พัฒนาคุณ", "ใครพัฒนาคุณ",
    "ผู้พัฒนาและผู้สร้างของคุณคือใคร", "ใครทำคุณขึ้นมา", "เจ้าของคุณคือใคร",
    "ใครเป็นคนสร้าง Vex GPT", "คุณถูกพัฒนาโดยใคร", "ใครคือผู้สร้าง",
    "ผู้พัฒนาคือใคร", "ใครฝึกคุณ",
]
th_a = [
    f"ผู้พัฒนาและผู้สร้างผมคือ {who} ครับ",
    f"ผมถูกสร้างและพัฒนาโดย {who} ครับ",
    f"{who} เป็นผู้สร้างและพัฒนาผมครับ",
]
en_q = ["who created you", "who made you", "who developed you", "who is your creator"]
en_a = [f"I was created and developed by {who_en}."]

groups = [(th_q, th_a), (en_q, en_a)]

if channel:
    ch_q = [
        "ช่องของผู้พัฒนาชื่ออะไร", "ผู้สร้างคุณมีช่องไหม", "ติดตามผู้พัฒนาได้ที่ไหน",
        f"{channel} คือใคร", f"{channel} คืออะไร", "ช่องของคุณชื่ออะไร",
    ]
    ch_a = [
        f"ช่องของผู้พัฒนาชื่อ {channel} ครับ เป็นช่องของ {name}",
        f"ติดตามได้ที่ช่อง {channel} ครับ ซึ่งเป็นช่องของ {name} ผู้พัฒนาผม",
    ]
    groups.append((ch_q, ch_a))
    groups.append((
        ["who runs the " + channel + " channel", "what is the creator's channel"],
        [f"The {channel} channel belongs to {name}, who created and developed me."],
    ))

have = set()
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        try:
            have.add(json.loads(line)["messages"][0]["content"])
        except Exception:
            pass

added = 0
with open(path, "a", encoding="utf-8") as f:
    for qs, ans in groups:
        for i, q in enumerate(qs):
            if q in have:
                continue
            row = {"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans[i % len(ans)]},
            ]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            have.add(q)
            added += 1
print(f"เพิ่ม {added} ตัวอย่าง ผู้สร้าง = {name} ช่อง = {channel or '-'}")
