"""Tag notebook cells with RISE slide types based on header level.

Rules:
    - empty source       -> "skip"
    - markdown '# '      -> "slide"
    - markdown '## '     -> "slide"
    - markdown '### '    -> "subslide"
    - everything else    -> unchanged (no tag = flows with current slide)

Usage:
    uv run python scripts/tag_slides.py 1_basics.ipynb 2_pytorch_fit.ipynb ...
    uv run python scripts/tag_slides.py *.ipynb
"""
import sys
import nbformat


def _source_text(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.strip()


def tag(path):
    nb = nbformat.read(path, as_version=4)
    counts = {"slide": 0, "subslide": 0, "skip": 0, "untagged": 0}

    for cell in nb.cells:
        text = _source_text(cell)
        meta = cell.setdefault("metadata", {})
        slideshow = meta.setdefault("slideshow", {})

        if not text:
            slideshow["slide_type"] = "skip"
            counts["skip"] += 1
            continue

        if cell.cell_type == "markdown":
            if text.startswith("# "):
                slideshow["slide_type"] = "slide"
                counts["slide"] += 1
                continue
            if text.startswith("## "):
                slideshow["slide_type"] = "slide"
                counts["slide"] += 1
                continue
            if text.startswith("### "):
                slideshow["slide_type"] = "subslide"
                counts["subslide"] += 1
                continue

        # Otherwise: explicit empty string = "default / flow with current slide"
        slideshow["slide_type"] = ""
        counts["untagged"] += 1

    nbformat.write(nb, path)
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"{path}: {summary}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: tag_slides.py notebook1.ipynb [notebook2.ipynb ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        tag(p)
