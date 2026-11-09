# search.py — ค้นเว็บด้วย DuckDuckGo (ฟรี ไม่ต้องใช้ API key)
#   ติดตั้ง: !pip install ddgs

def web_search(query, n=4):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    try:
        with DDGS() as d:
            results = list(d.text(query, max_results=n))
    except Exception as e:
        return f"(ค้นหาไม่สำเร็จ: {e})"
    if not results:
        return "(ไม่พบผลลัพธ์)"
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
        for i, r in enumerate(results, 1)
    )


def build_prompt(question):
    ctx = web_search(question)
    return (
        "ใช้ข้อมูลจากผลค้นหาด้านล่างช่วยตอบคำถาม "
        "ถ้าข้อมูลไม่พอให้บอกตรง ๆ ห้ามเดา\n\n"
        f"ผลค้นหา:\n{ctx}\n\nคำถาม: {question}"
    )
