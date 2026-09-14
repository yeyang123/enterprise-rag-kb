"""
Week3-Day1：文档加载器第一步 —— PDF 文本提取
pdfplumber 逐页模型：open() 得到 PDF 对象，.pages 是页列表，每页 extract_text()
"""
import pdfplumber

def extract_pdf_text(path: str) -> list[dict]:
    """
    提取 PDF 全文。
    返回【按页】的字典列表，而不是直接拼一个大字符串——
    因为"页码"是 Day5 引用溯源的关键元数据，现在丢掉以后就找不回来了。
    返回示例：[{"page": 1, "text": "第一页文字..."}, ...]
    """
    pages = []
    # with 语句保证文件句柄被关闭（和 Day3 读 TXT 的 open 同理）
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):  # 页码从 1 开始，符合人的习惯
            text = page.extract_text() or ""           # 扫描页/空白页可能返回 None，兜底成 ""
            pages.append({"page": i, "text": text})
    return pages

if __name__ == "__main__":
    # 最小自检：命令行参数传 PDF 路径
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"data\raw\test.pdf"
    result = extract_pdf_text(pdf_path)

    print(f"共 {len(result)} 页\n")
    for p in result[:3]:                          # 先只预览前 3 页，避免刷屏
        preview = p["text"][:200].replace("\n", " ")
        print(f"--- 第 {p['page']} 页（{len(p['text'])} 字）---")
        print(preview + "\n")

    # 扫描版自检提示：如果每页都是 0 字，大概率是图片型 PDF，需要 OCR
    total = sum(len(p["text"]) for p in result)
    if total == 0:
        print("⚠️ 没提取到任何文字：可能是扫描版（图片）PDF，需要 OCR 方案")