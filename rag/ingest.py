"""
Week3-Day6：批量 ingest（Week3 集成日）
输入：data/raw/ 下所有 PDF/TXT
流程：load_document -> chunk_document -> 写 data/chunks.jsonl
特性：
  1) 幂等：以 "source" 区分文件，chunks.jsonl 里只保留本次扫描到的文件
  2) 质检：打印每文档 chunk 数、长度分布、硬切超长块占比
"""
import json
import os

from rag.loader import load_document
from rag.chunker import chunk_document

RAW_DIR = "data/raw"
OUT_FILE = "data/chunks.jsonl"
SUPPORTED = (".pdf", ".txt")

def list_documents(raw_dir: str) -> list[str]:
    """列出原始目录下所有受支持文件（按文件名排序，保证输出稳定）。"""
    files = []
    for name in sorted(os.listdir(raw_dir)):
        if name.lower().endswith(SUPPORTED):
            files.append(os.path.join(raw_dir, name))
    return files

def ingest_file(path: str) -> list[dict]:
    """单文件：加载 -> 切块。异常向上抛由调用方决定跳过还是中断。"""
    doc =load_document(path)
    return chunk_document(doc)

def write_jsonl(path: str, chunks: list[dict]):
    """整体重写输出文件（最简单可靠的幂等：以本次扫描结果为准）。
    JSONL：每行一个完整 JSON 对象，ensure_ascii=False 保留中文。"""
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def stats(chunks: list[dict], chunk_size: int = 500) -> dict:
    """计算长度分布等质检指标。"""
    lens = sorted(c["char_count"] for c in chunks)
    n = len(lens)
    # 超长块：长度达到 chunk_size 上限的，几乎都走过硬切兜底分支
    hard_cut = sum(1 for x in lens if x >= chunk_size)
    return {
        "total": n,
        "min": lens[0] if n else 0,
        "p50": lens[n // 2] if n else 0,           # 中位数
        "avg": sum(lens) // n if n else 0,
        "max": lens[-1] if n else 0,
        "hard_cut_ratio": f"{hard_cut / n:.0%}" if n else "-",
    }

def main():
    files = list_documents(RAW_DIR)
    if not files:
        print(f"⚠️ {RAW_DIR} 下没有 PDF/TXT 文件")
        return

    all_chunks, current_sources = [], set()
    per_doc = []

    for path in files:
        name = os.path.basename(path)
        try:
            chunks = ingest_file(path)
        except Exception as e:
            # 单文件坏掉不拖垮整批：打印错误后跳过（注意：失败文件不会进输出）
            print(f"❌ 跳过 {name}：{e}")
            continue
        all_chunks.extend(chunks)
        current_sources.add(name)
        s = stats(chunks)
        per_doc.append((name, s))

    write_jsonl(OUT_FILE, all_chunks)

    # ---- 质检报告 ----
    print("=" * 66)
    print(f"{'文件':<22}{'块数':>5}{'最短':>5}{'中位':>5}{'均值':>5}{'最长':>6}{'硬切%':>7}")
    print("-" * 66)
    for name, s in per_doc:
        print(f"{name:<22}{s['total']:>5}{s['min']:>5}{s['p50']:>5}"
              f"{s['avg']:>5}{s['max']:>6}{s['hard_cut_ratio']:>7}")
    print("-" * 66)
    total_s = stats(all_chunks)
    print(f"合计 {len(current_sources)} 个文件 / {total_s['total']} 个 chunk"
          f"（中位 {total_s['p50']} / 均值 {total_s['avg']} / 硬切 {total_s['hard_cut_ratio']}）")
    print(f"已写出：{OUT_FILE}")

    # 幂等自检：输出里的 source 集合应恰好等于本次成功处理的文件集合
    sources_in_file = {c["source"] for c in all_chunks}
    assert sources_in_file == current_sources, "来源集合不一致，幂等检查失败"
    print("✅ 幂等自检通过")

if __name__ == "__main__":
    main()