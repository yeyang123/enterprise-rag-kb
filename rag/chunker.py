"""
Week3-Day3：朴素定长切块
text[i:i+chunk_size] 纯位置切片——今天故意保留它的缺陷，
亲眼观察句子被拦腰切断的 badcase，D4 用句边界 + overlap 修复。
复用 Day2 的 load_document() 拿清洗后的分页文本。
"""
import re
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

def split_sentences(text: str) -> list[str]:
    """
    按句末标点切句，标点保留在句尾。
    (?<=...) 是"后行断言"：在标点【之后】下刀，标点归前一句。
    """
    parts = re.split(r"(?<=[。！？；!?;…])", text)
    return [s for s in (p.strip() for p in parts) if s]

def overlap_chunk(text: str, chunk_size: int = 500, overlap: int = 80) -> list[tuple]:
    """句边界+重叠切块。返回 [(chunk文本, 在原文的起点, 终点), ...]
    位置信息在切块时一并记录——overlap 会让位置非线性，事后无法靠乘法推算。"""
    sents = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    # pos：句子在原文中的定位游标，用 find(..., pos) 顺序搜索（允许句子重复出现）
    pos = 0

    def locate(s):
        nonlocal pos
        idx = text.find(s, pos)
        pos = idx + len(s)
        return idx

    for s in sents:
        s_start = locate(s)
        while len(s) > chunk_size:   # 超长句硬切，同时记录位置
            if cur:
                chunks.append(("".join(t[0] for t in cur), cur[0][1], cur[-1][2]))
                cur, cur_len = [], 0
            chunks.append((s[:chunk_size], s_start, s_start + chunk_size))
            s, s_start = s[chunk_size:], s_start + chunk_size

        if cur and cur_len + len(s) > chunk_size:
            chunks.append(("".join(t[0] for t in cur), cur[0][1], cur[-1][2]))
            tail, t_len = [], 0
            for prev in reversed(cur):
                if t_len >= overlap:
                    break
                tail.insert(0, prev)
                t_len += len(prev[0])
            cur, cur_len = tail, t_len

        cur.append((s, s_start, s_start + len(s)))
        cur_len += len(s)

    if cur:
        chunks.append(("".join(t[0] for t in cur), cur[0][1], cur[-1][2]))
    return chunks

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
    for i, (text, start, end) in enumerate(overlap_chunk(full_text, chunk_size)):
        pages = sorted({m for m in page_marks[start:end] if m is not None})
        chunks.append({
            "chunk_id": f"{doc['source']}#chunk-{i:04d}",
            "index": i,
            "text": text,
            "source": doc["source"],
            "pages": pages,                                   # 精确页码列表
            "page_start": pages[0] if pages else None,        # 保留便捷字段
            "page_end": pages[-1] if pages else None,
            "char_count": len(text),
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
              f"（第 {c['pages']} 页，{len(c['text'])} 字）=====")
        print(f"[开头] {c['text'][:60]}")
        print(f"[结尾] {c['text'][-60:]}")
        print()