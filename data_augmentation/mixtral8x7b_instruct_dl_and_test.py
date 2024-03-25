from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

text = "Hello my name is"
messages = [
    {"role": "user", "content": "What is your favorite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavor to whatever I'm cooking in the kitchen."},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=2000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
