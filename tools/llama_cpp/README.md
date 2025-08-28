Vendored llama.cpp converter

- Path: tools/llama_cpp/convert-hf-to-gguf.py
- Copy the upstream file from:
  https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py
- Configure your exporter to point [export.ollama].convert_bin to the absolute path of the file above.
- Verify: uv run python tools/llama_cpp/convert-hf-to-gguf.py --help
