# Llama.cpp converter (vendored)

- Path: tools/llama_cpp/convert-hf-to-gguf.py
- Copy the upstream file from:
  [raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py](https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py)
- Configure your exporter to point \[export.ollama\].convert_bin to the absolute path of the file above.
- Verify: `make gguf-help`

## Folder structure

```text
tools/llama_cpp/
├── README.md        - usage notes and how to vendor the converter
├── llama-gguf       - upstream binary or helper (if present)
└── llama-quantize   - upstream binary or helper (if present)
```
