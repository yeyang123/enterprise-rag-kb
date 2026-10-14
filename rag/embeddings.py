"""
Week4-Day1：Embedding 客户端（本地 + API 双后端，统一接口）
- local: sentence-transformers + BAAI/bge-small-zh-v1.5（512 维，免费离线）
- api:   硅基流动 BAAI/bge-large-zh-v1.5（1024 维，OpenAI 兼容格式）
铁律：同一个 Chroma collection 只能用同一个模型，维度不同不可混用。
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
API_MODEL = "BAAI/bge-large-zh-v1.5"
API_URL = "https://api.siliconflow.cn/v1/embeddings"

class EmbeddingClient:
    """统一接口：embed_texts(list[str]) -> list[list[float]]
    业务代码只认这个方法，不关心背后是本地模型还是 HTTP API。"""

    def __init__(self, backend: str | None = None):
        self.backend = backend or os.getenv("EMBED_BACKEND", "local")

        if self.backend == "local":
            # 延迟导入：选 api 后端的机器不用装 torch
            from sentence_transformers import SentenceTransformer
            # 首次运行会下载模型(~100MB)到用户缓存目录；下载慢见计划里的 HF 镜像说明
            self.model = SentenceTransformer(LOCAL_MODEL)
            self.dim = self.model.get_sentence_embedding_dimension()
        elif self.backend == "api":
            self.key = os.getenv("SILICONFLOW_API_KEY")
            if not self.key:
                raise ValueError(".env 里缺 SILICONFLOW_API_KEY")
            self.dim = 1024  # bge-large-zh 的固定维度
        else:
            raise ValueError(f"未知后端：{self.backend}（只支持 local / api）")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.backend == "local":
            # normalize=True：向量归一化成单位长度，之后内积=余弦相似度（D2 会用到）
            vecs = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 50,
            )
            return [v.tolist() for v in vecs]

        # api：OpenAI 兼容格式，注意分批（每批 16 条，防止单次请求过大）
        out = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.key}"},
                json={
                    "model": API_MODEL,
                    "input": batch,
                    "encoding_format": "float",
                },
                timeout=60,
            )
            resp.raise_for_status()
            # 响应里 data 的顺序不保证，按 index 字段排回原顺序
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
            print(f"[api] 已向量化 {min(i + 16, len(texts))}/{len(texts)}")
        return out

if __name__ == "__main__":
    client = EmbeddingClient()
    texts = [
        "电池续航多久",             # A
        "充电一次能用多长时间",       # B（和 A 同义，字面零重叠）
        "今天天气不错",             # C（无关句）
    ]
    vecs = client.embed_texts(texts)
    print(f"后端: {client.backend} | 向量数: {len(vecs)} | 维度: {len(vecs[0])}")
    print("第一句前 5 维:", [round(x, 4) for x in vecs[0][:5]])