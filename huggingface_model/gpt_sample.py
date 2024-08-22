from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2-custom")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-custom")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(output)