from transformers import AutoTokenizer, AutoConfig, AutoModel

config = AutoConfig.from_pretrained("gpt2-custom")
config.push_to_hub("custom_gpt2")
model = AutoModel.from_pretrained("gpt2-custom")
model.push_to_hub("custom_gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2-custom")
tokenizer.push_to_hub("custom_gpt2")