# 企业文档 RAG 问答知识库

基于 RAG（检索增强生成）的私有文档智能问答系统。支持上传 PDF/TXT，自动解析、分块、向量化入库，基于语义检索 + DeepSeek 生成答案。

## 技术栈

- **文档解析**：pdfplumber（文本提取 + 表格识别）
- **文本分块**：句边界打包 + overlap 重叠分块
- **Embedding**：BAAI/bge-small-zh-v1.5（本地 512 维）/ 硅基流动 API（1024 维）
- **向量库**：Chroma（本地持久化，无需部署）
- **LLM 调用**：DeepSeek API（流式输出 + 超时重试）
- **前端**：Streamlit

## 项目结构

```text
enterprise-rag-kb/
├─ llm_gateway/          # LLM 调用层（流式/重试/异常，复用自阶段1）
│  ├─ http_client.py     # requests 裸调 + 指数退避
│  └─ llm_client.py      # 流式/非流式统一封装
├─ rag/
│  ├─ loader.py          # 文档解析：PDF/TXT 清洗、表格提取、目录过滤
│  ├─ chunker.py         # 文本分块：句边界 + overlap
│  ├─ ingest.py          # 批量摄取：文件夹 → chunks.jsonl（幂等）
│  ├─ embeddings.py      # Embedding 客户端（本地/API 双后端）
│  ├─ vectorstore.py     # Chroma 入库与查询（Week4 待建）
│  ├─ retriever.py       # 相似度召回（Week4 待建）
│  └─ chain.py           # RAG 完整链路（Week5 待建）
├─ data/
│  ├─ raw/               # 原始 PDF/TXT（.gitignore）
│  └─ chunks.jsonl       # 分块结果（.gitignore）
├─ app.py                # Streamlit 前端（Week5 待建）
├─ cli.py                # 命令行问答（Week5 待建）
└─ eval/
└─ questions.csv      # 召回评估问题集（Week6 待建）

## 快速开始

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（.env 文件）
DEEPSEEK_API_KEY=sk-xxx
EMBED_BACKEND=local   # 或 api（硅基流动）

# 3. 把 PDF 放进 data/raw/，跑摄取
python -m rag.ingest


## 开发进度

- [x] Week3：文档解析（pdfplumber）+ 文本清洗 + 分块
- [x] Week4 Day1：Embedding 双后端封装（本地 bge / API）
- [ ] Week4 Day2-6：向量相似度实验 → Chroma 入库 → top-k 召回 → 人工评测
- [ ] Week5：RAG 完整链路（CLI + Streamlit 前端）
- [ ] Week6：混合检索 + Rerank + 幻觉抑制 + 项目包装