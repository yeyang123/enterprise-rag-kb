import json
import os
import time  # 新增：重试时 sleep 用

import requests
from dotenv import load_dotenv

load_dotenv()

# DeepSeek 的 OpenAI 兼容接口：SDK 底下其实就是 POST 到这个 URL
API_URL = "https://api.deepseek.com/chat/completions"

class LLMCallError(Exception):
    """业务层统一异常：超时/断网/4xx/5xx/JSON 损坏/结构异常，全都包成它。
    调用方只需 try/except LLMCallError 一种，不用关心 requests 的异常家族。"""

def _is_retryable_status(code: int) -> bool:
    """判断 HTTP 状态码是否可重试：429（限流）和 5xx（服务器内部错误）可以重试。
    4xx（客户端错误，如 400/401/403/404）不重试——你的请求本身有问题，重试也没用。"""
    return code == 429 or 500 <= code < 600

def chat_raw(messages: list[dict], temperature: float = 0.1, timeout: int = 30,
             max_retries: int = 3) -> dict:
    """
    裸调 HTTP（带超时 + 异常处理 + 指数退避重试）。
    max_retries=3 表示最多尝试 3 次，失败后等 1s → 2s → 再放弃。
    任何最终失败都包装成 LLMCallError 抛出。
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
    }

    last_error = None  # 记录最后一次异常，重试耗尽后用它构造错误消息

    # range(max_retries) = [0, 1, 2]，共尝试 3 次
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()          # 4xx/5xx 抛 HTTPError；2xx 继续
            return resp.json()              # ✅ 成功，直接返回（跳出循环）

        # ---- 可重试错误 1：超时 ----
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt            # 指数退避：1s, 2s
                print(f"[重试 {attempt+1}/{max_retries}] ⏱ 超时，{wait}s 后重试...")
                time.sleep(wait)
                continue
            # 最后一次仍超时，往下走抛出

        # ---- 可重试错误 2：连接失败 ----
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[重试 {attempt+1}/{max_retries}] 🔌 连接失败，{wait}s 后重试...")
                time.sleep(wait)
                continue

        # ---- HTTP 错误：429/5xx 可重试，4xx 不重试 ----
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if _is_retryable_status(status):
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[重试 {attempt+1}/{max_retries}] HTTP {status}，{wait}s 后重试...")
                    time.sleep(wait)
                    continue
            # 4xx（非429）：不重试，直接报错（重试也没用）
            raise LLMCallError(
                f"❌ HTTP {status}（不重试）：{e.response.text[:300]}"
            ) from e

        # ---- 不可重试：JSON 解析失败（服务器返回了非 JSON，重试大概率还是错）----
        except requests.exceptions.JSONDecodeError as e:
            raise LLMCallError(f"📦 响应不是合法 JSON：{resp.text[:200]}") from e

    # 循环耗尽仍没成功 → 抛出最后一次的错误
    raise LLMCallError(
        f"❌ 重试 {max_retries} 次后仍失败：{last_error}"
    ) from last_error

def chat_with_messages(messages: list[dict], temperature: float = 0.1, timeout: int = 30,
                       max_retries: int = 3):
    """返回 (answer, usage)。结构异常也包成 LLMCallError。"""
    data = chat_raw(messages, temperature=temperature, timeout=timeout, max_retries=max_retries)
    try:
        # KeyError：缺键；IndexError：choices 是空列表；TypeError：结构类型不对
        answer = data["choices"][0]["message"]["content"]
        usage = data["usage"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMCallError(f"🧩 响应结构异常，取不到回答内容：{data}") from e
    return answer, usage

def ask_llm(prompt: str, system: str = "", temperature: float = 0.1, timeout: int = 30,
            max_retries: int = 3):
    """单轮提问便捷封装（Day6），透传 timeout 与 max_retries。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat_with_messages(messages, temperature=temperature, timeout=timeout,
                              max_retries=max_retries)

if __name__ == "__main__":
        # ============================================================
    # 【Day6 实验】同一问题，三个温度对比 + token 观察
    # ============================================================
    PROMPT = "用一句话介绍 Python 的列表推导式"
    SYSTEM = "你是一个简洁的 Python 老师"

    print("=" * 60)
    print("【Day6】温度对比实验")
    print("=" * 60)

    for t in (0.1, 0.7, 1.5):  # 低 / 中 / 高
        ans, usage = ask_llm(PROMPT, system=SYSTEM, temperature=t)
        print(f"\n--- temperature = {t} ---")
        print(f"回答: {ans}")
        print(f"token: 输入 {usage['prompt_tokens']} / "
              f"输出 {usage['completion_tokens']} / "
              f"合计 {usage['total_tokens']}")
    # ============================================================
    # 第 1 部分：单次调用，打印完整响应 JSON
    # ============================================================
    msgs = [{"role": "user", "content": "什么是HTTP？一句话回答"}]

    data = chat_raw(msgs)

    print("=" * 60)
    print("【1】完整响应 JSON")
    print("=" * 60)
    # ensure_ascii=False 中文正常显示；indent=2 缩进排版
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # ============================================================
    # 第 2 部分：层层拆包，对照上面的 JSON 看取值路径
    # ============================================================
    print("=" * 60)
    print("【2】一层层取值")
    print("=" * 60)
    print(f"data                    类型: {type(data).__name__}")                 # dict
    print(f"data['choices']         类型: {type(data['choices']).__name__}")     # list
    print(f"data['choices'][0]      类型: {type(data['choices'][0]).__name__}")  # dict
    print(f"data['choices'][0]['message'] 内容: {data['choices'][0]['message']}")

    # 沿着结构一路取到底：字典 -> 列表 -> 字典 -> 字典
    content = data["choices"][0]["message"]["content"]
    print(f"\n最终答案: {content}")

    # usage 也是字典（SDK 里是对象 res.usage.prompt_tokens）
    print(f"输入 token: {data['usage']['prompt_tokens']}")
    print(f"输出 token: {data['usage']['completion_tokens']}")

    # 对照记忆（本质是同一份数据，SDK 只是把 resp.json() 的字典包装成了对象）：
    # SDK : res.choices[0].message.content          （点号取属性）
    # 裸调: data["choices"][0]["message"]["content"] （方括号取键 / 下标）

    # ============================================================
    # 第 3 部分：多轮对话（复刻 main.py，换用裸调客户端）
    # ============================================================
    print("=" * 60)
    print("【3】多轮对话（输入 exit 退出）")
    print("=" * 60)

    messages = [{"role": "system", "content": "你是一个简洁、耐心的 Python 学习助手，回答尽量简短。"}]

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        try:
            answer, usage = chat_with_messages(messages)
        except Exception as e:
            print(f"❌ 调用失败：{e}")
            messages.pop()  # 回滚没得到回复的提问
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"\n助手：{answer}")
        print(f"[输入 {usage['prompt_tokens']} tok / 输出 {usage['completion_tokens']} tok]")

    print("\n👋 对话结束，再见！")