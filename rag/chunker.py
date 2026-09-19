"""
Week3-Day3：朴素定长切块
text[i:i+chunk_size] 纯位置切片——今天故意保留它的缺陷，
亲眼观察句子被拦腰切断的 badcase，D4 用句边界 + overlap 修复。
复用 Day2 的 load_document() 拿清洗后的分页文本。
"""
from rag.loader import load_document

def naive_chunk(text: str, chunk_size: int = 500) -> list[str]:
    """
    最朴素的切块：每 chunk_size 个字符切一刀。
    已知缺陷（今天要观察的）：
      1) 句子被拦腰切断（chunk 尾/头各半句）
      2) 词可能被切成两半
      3) 相邻 chunk 之间零重叠，跨块的信息两头都残缺
    """
    text = text.strip()
    if not text:
        return []
    # range(0, len, step) 生成每个切块的起点：0, 500, 1000, ...
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_document(doc: dict, chunk_size: int = 500) -> list[dict]:
    """
    对一整个文档切块。先把各页文本拼起来再切——
    因为一个句子/主题经常跨页，按页切会人为制造断点。
    返回 [{text, source, page_start, page_end}, ...]
    （页码信息先简单记录起止，Day5 再做精确到 chunk 的元数据）
    """
    # 拼接时记录每个字符属于哪一页：page_marks[i] = 第 i 个字符的页码
    full_text = ""
    page_marks = []
    for p in doc["pages"]:
        t = p["text"]
        full_text += t + "\n"
        page_marks.extend([p["page"]] * (len(t) + 1))

    chunks = []
    for i, text in enumerate(naive_chunk(full_text, chunk_size)):
        start = i * chunk_size                       # 该块起点在 full_text 里的位置
        end = start + len(text) - 1
        pages = sorted({m for m in page_marks[start:end] if m is not None})
        chunks.append({
            "text": text,
            "source": doc["source"],
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
        })
    return chunks

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"data\raw\test.pdf"
    doc = load_document(path)
    chunks = chunk_document(doc, chunk_size=500)

    print(f"文档 {doc['source']}：{sum(len(p['text']) for p in doc['pages'])} 字"
          f" -> {len(chunks)} 个 chunk（500 字/块）\n")

    # 重点观察：每个 chunk 的【开头】和【结尾】——
    # 开头大概率是半句话，结尾大概率切在句子中间
    for c in chunks[:5]:
        print(f"===== chunk {c['chunk_id'] if 'chunk_id' in c else chunks.index(c)}"
              f"（第 {c['page_start']}-{c['page_end']} 页，{len(c['text'])} 字）=====")
        print(f"[开头] {c['text'][:60]}")
        print(f"[结尾] {c['text'][-60:]}")
        print()