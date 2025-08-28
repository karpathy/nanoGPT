#!/usr/bin/env python3
"""
Vendored path for llama.cpp's convert-hf-to-gguf.py.

Action required:
- Replace this file's contents with the official script from:
  https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py

This placeholder exists so tools can be configured to a stable path
(tools/llama_cpp/convert-hf-to-gguf.py). If executed as-is, it will
print a helpful message and exit.

Upstream license applies to the real script once you replace this file.
"""

import sys
import textwrap


def main() -> int:
    msg = textwrap.dedent(
        """
        Placeholder: convert-hf-to-gguf.py not yet vendored.

        Please copy the upstream llama.cpp converter into:
          tools/llama_cpp/convert-hf-to-gguf.py

        Fetch from:
          https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py

        Then re-run your command, or invoke:
          uv run python tools/llama_cpp/convert-hf-to-gguf.py --help
        """
    ).strip()
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
