# finetune.py — Fine-tune Qwen2.5-Coder ให้คุยไทย + เขียนโค้ด (Unsloth + LoRA)
#
# ใช้บน Google Colab (GPU T4):
#   !pip install unsloth
#   อัปโหลด data.jsonl แล้วรัน: !python finetune.py
#
# รูปแบบ data.jsonl (1 บรรทัด = 1 ตัวอย่าง):
#   {"messages":[{"role":"user","content":"เขียนฟังก์ชันหาเลขคู่"},
#                {"role":"assistant","content":"```python\ndef evens(xs): ...\n```"}]}

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # เปลี่ยนเป็น 7B-Instruct ได้ถ้า GPU พอ
DATA = "data.jsonl"
OUT = "lora-th-coder"
MAX_LEN = 2048

model, tok = FastLanguageModel.from_pretrained(
    MODEL, max_seq_length=MAX_LEN, load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

ds = load_dataset("json", data_files=DATA, split="train")
# ใส่บุคลิกจาก persona.txt เป็น system prompt ในทุกตัวอย่าง (ให้ตรงกับตอนคุยใน chat.py)
import os

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "persona.txt"), encoding="utf-8") as _f:
    PERSONA = _f.read().strip()


def add_persona(x):
    m = x["messages"]
    if m and m[0]["role"] != "system":
        m = [{"role": "system", "content": PERSONA}] + m
    return {"text": tok.apply_chat_template(m, tokenize=False)}


ds = ds.map(add_persona)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir="out",
        report_to="none",
    ),
)

trainer.train()

model.save_pretrained(OUT)
tok.save_pretrained(OUT)

# แปลงเป็น GGUF สำหรับ Ollama / llama.cpp (เอา comment ออกถ้าต้องการ)
# model.save_pretrained_gguf("gguf", tok, quantization_method="q4_k_m")
