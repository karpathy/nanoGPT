from llama_cpp import Llama

text = """<s>[INST] What is your favourite condiment? [/INST]
Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s>
[INST] Do you have mayonnaise recipes? [/INST]"""

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf", n_ctx=2048,
            n_threads=8, n_gpu_layers=35, verbose=True)

output = llm(text, max_tokens=256, stop=["[INST]"], echo=True)

print(output)
