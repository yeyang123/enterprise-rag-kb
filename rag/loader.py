"""
Week3-Day2：统一文档加载器 + 文本清洗 + 表格提取
在 D1 裸提取基础上增加：
  1) clean_page_text：修硬换行/多余空白/全角空格
  2) find_repeated_lines：跨页重复行识别页眉页脚
  3) _tables_to_text：extract_tables() 二维列表转可读文本
  4) load_document：PDF/TXT 统一入口，下游只认一种数据结构
"""
import os
import re
from collections import Counter

import pdfplumber

# 句末标点：行尾遇到它们，说明段落真的结束了，换行要保留
_SENTENCE_END = "。！？；…!?;…"

def _both_cjk(a: str, b: str) -> bool:
    """两个字符是否都是中文（决定硬换行拼接时要不要补空格）。"""
    return bool(re.match(r"[\u4e00-\u9fff]", a) and re.match(r"[\u4e00-\u9fff]", b))

def clean_page_text(text: str) -> str:
    """清洗单页文本：解决 PDF 提取最典型的脏数据。"""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]

    merged = []
    for line in lines:
        if not line:
            # 空行可能是段落间隔：多个连续空行压成一个
            if merged and merged[-1] != "":
                merged.append("")
            continue

        line = line.replace("\u3000", " ")           # 全角空格转半角
        line = re.sub(r"[ \t]+", " ", line).strip()  # 多个空格压成一个

        if not merged or merged[-1] == "":
            merged.append(line)
            continue

        prev = merged[-1]
        if prev[-1] in _SENTENCE_END:
            # 上一行以句末标点结尾 -> 段落结束，保留换行
            merged.append(line)
        else:
            # 硬换行：两行本是一句话，拼起来
            # 中文+中文直接拼；涉及英文/数字补空格，防止 "errorcode" 粘连
            gap = "" if _both_cjk(prev[-1], line[0]) else " "
            merged[-1] = prev + gap + line

    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 连续空行最多保留一个
    return text.strip()

def find_repeated_lines(pages_text: list[str], min_ratio: float = 0.7) -> set[str]:
    """
    识别页眉页脚：在足够多页面里【逐字相同】的行。
    原理：正文几乎不会跨页原样重复，页眉页脚会。
    min_ratio：出现页数占比阈值；另外纯数字短行（页码，如 "12"）直接收掉。
    """
    counter = Counter()
    for text in pages_text:
        # 每页只计一次（集合去重），防止同一行在一页内重复刷高计数
        unique = {ln.strip() for ln in text.split("\n") if ln.strip()}
        for ln in unique:
            counter[ln] += 1

    threshold = max(2, int(len(pages_text) * min_ratio))  # 至少 2 页且占比 ≥70%
    return {
        ln for ln, cnt in counter.items()
        if cnt >= threshold or (ln.isdigit() and len(ln) <= 3)
    }

def _tables_to_text(page) -> str:
    """把一页内的所有表格转成可读文本；没有表格返回空串。"""
    chunks = []
    for table in page.extract_tables() or []:
        rows = []
        for row in table:
            # 单元格 None -> ""；格子内的换行压成空格
            cells = [(c or "").replace("\n", " ").strip() for c in row]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            chunks.append("【表格】\n" + "\n".join(rows))
    return "\n".join(chunks)

def extract_pdf_text(path: str) -> list[dict]:
    """提取 PDF：逐页拿正文 + 表格，先单页清洗，再跨页删页眉页脚。"""
    raw_pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            table_text = _tables_to_text(page)
            if table_text:
                text = f"{text}\n{table_text}" if text.strip() else table_text
            raw_pages.append({"page": i, "text": text})

    # 顺序不能反：先清洗单页，再跨页找重复行
    for p in raw_pages:
        p["text"] = clean_page_text(p["text"])

    repeated = find_repeated_lines([p["text"] for p in raw_pages])
    if repeated:
        print(f"[清洗] 识别并删除页眉页脚 {len(repeated)} 行：{list(repeated)[:3]} ...")

    cleaned_pages = []
    for p in raw_pages:
        lines = [ln for ln in p["text"].split("\n") if ln.strip() not in repeated]
        p["text"] = "\n".join(lines).strip()
        if p["text"]:  # 清洗后完全空白的页（纯分隔页）不进下游
            cleaned_pages.append(p)
    return cleaned_pages

def extract_txt_text(path: str) -> list[dict]:
    """TXT 没有页码概念，统一归为第 1 页，保证下游数据结构一致。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = clean_page_text(text)
    return [{"page": 1, "text": text}] if text else []

def load_document(path: str) -> dict:
    """
    统一加载入口。PDF/TXT 都返回同一种结构：
    {"source": 文件名, "doc_type": "pdf"/"txt",
     "pages": [{"page": 页码, "text": "清洗后文本"}, ...]}
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = extract_pdf_text(path)
    elif ext == ".txt":
        pages = extract_txt_text(path)
    else:
        raise ValueError(f"暂不支持的文件类型：{ext}（目前只支持 .pdf / .txt）")

    return {
        "source": os.path.basename(path),
        "doc_type": ext.lstrip("."),
        "pages": pages,
    }

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"data\raw\test.pdf"
    doc = load_document(path)

    pages = doc["pages"]
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"文件：{doc['source']}（{doc['doc_type']}）")
    print(f"有效页数：{len(pages)}，总字符：{total_chars}\n")

    for p in pages[:2]:
        print(f"===== 第 {p['page']} 页（{len(p['text'])} 字）=====")
        print(p["text"][:400] + "\n")