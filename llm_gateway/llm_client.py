"""
统一 LLM 客户端（Week2-Day5 集成版）
- 非流式：复用 http_client.chat_raw（已含 timeout + 指数退避重试 + LLMCallError）
- 流式：stream_chat_text() 手写 SSE，连接阶段可重试，输出中途断开不重试
对外保持原有接口：chat_with_messages / get_llm_response（main.py 不受影响）
"""
import json
import os
import time

import requests
from dotenv import load_dotenv

from .http_client import API_URL, LLMCallError, chat_raw  # 复用 W2D3/W2D4 的传输层

load_dotenv()

# ============================================================
# 非流式：直接转发给 http_client，不重复造轮子
# ============================================================
def chat_with_messages(messages: list[dict], temperature: float = 0.1,
                       timeout: int = 30, max_retries: int = 3):
    """多轮消息调用，返回 (answer, usage)。超时/重试/异常全由 chat_raw 负责。"""
    data = chat_raw(messages, temperature=temperature, timeout=timeout, max_retries=max_retries)
    try:
        answer = data["choices"][0]["message"]["content"]
        usage = data["usage"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMCallError(f"🧩 响应结构异常，取不到回答内容：{data}") from e
    return answer, usage

def get_llm_response(prompt: str, system: str = "", temperature: float = 0.1,
                     timeout: int = 30, max_retries: int = 3):
    """单轮提问（Day2 老接口，保持兼容）。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat_with_messages(messages, temperature=temperature,
                              timeout=timeout, max_retries=max_retries)

# ============================================================
# 流式：生成器逐段 yield 文本增量（打字机效果）
# ============================================================
def stream_chat_text(messages: list[dict], temperature: float = 0.1,
                     connect_timeout: int = 10, read_timeout: int = 120,
                     max_retries: int = 3):
    """
    流式对话生成器：每次 yield 一小段文本。
    重试边界：连接阶段失败可重试；已经 yield 过内容后失败，直接抛 LLMCallError。
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "stream": True,                       # API 层：我要 SSE
    }
    last_error = None

    for attempt in range(max_retries):
        started = False   # 本轮是否已经向调用方吐过字（决定中途失败能不能重试）
        try:
            resp = requests.post(
                API_URL, headers=headers, json=payload,
                timeout=(connect_timeout, read_timeout),  # (连接超时, 数据间隔超时)
                stream=True,                              # requests 层：边收边读
            )
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue                             # 跳过 SSE 空行
                data = line[len("data: "):]
                if data == "[DONE]":
                    return                               # 正常结束，生成器停止
                chunk = json.loads(data)
                if not chunk.get("choices"):
                    continue                             # usage chunk 等无 choices 的包
                piece = chunk["choices"][0].get("delta", {}).get("content") or ""
                if piece:
                    started = True                       # 标记：已经开始输出了
                    yield piece                          # 把增量交给调用方打印
            return                                       # 流读完且未见 [DONE]，也算正常结束

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            # 4xx（429 除外）是你的请求有问题，重试无意义，直接报错
            if status != 429 and not (500 <= status < 600):
                raise LLMCallError(
                    f"❌ HTTP {status}（不重试）：{e.response.text[:300]}"
                ) from e
            last_error = e
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e

        # ---- 能走到这里说明本轮失败了，决定要不要重试 ----
        if started:
            # 已输出半截，重试会重复内容 -> 不重试
            raise LLMCallError(f"⛔ 流式输出中途中断（已开始输出，不重试）：{last_error}") \
                from last_error
        if attempt < max_retries - 1:
            wait = 2 ** attempt                          # 指数退避：1s, 2s
            print(f"\n[重试 {attempt + 1}/{max_retries}] "
                  f"{last_error.__class__.__name__}，{wait}s 后重试...")
            time.sleep(wait)
        else:
            raise LLMCallError(
                f"❌ 流式连接重试 {max_retries} 次后仍失败：{last_error}"
            ) from last_error