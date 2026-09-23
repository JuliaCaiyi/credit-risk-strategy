# 调用 Claude API 的最小封装（只用 Python 自带库，不需要额外安装）
# API key 放在项目根目录的 .llm_key 文件里（已被 .gitignore 忽略，不会上传 GitHub）
# 没有 key 时 available() 返回 False，各脚本自动走离线版本
import json
import os
import urllib.error
import urllib.request

MODEL = os.environ.get("RISK_LLM_MODEL", "claude-sonnet-5")
KEY_FILE = ".llm_key"


def _key():
    if os.path.exists(KEY_FILE):
        k = open(KEY_FILE).read().strip()
        return k or None
    return None


def available():
    return _key() is not None


def ask(prompt, system="", max_tokens=2000, temperature=0):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": _key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API 调用失败 {e.code}: {e.read().decode()[:500]}")
    return "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")


# 手动模式（没有 API key 时）：脚本把 prompt 存成文件，由人复制到 Claude 网页版的新对话里，
# 再把回答粘贴回 output/manual_llm/<name>.txt，脚本读取后照常校验
MANUAL_DIR = "output/manual_llm"


def save_prompt(name, prompt, system=""):
    os.makedirs(MANUAL_DIR, exist_ok=True)
    with open(f"{MANUAL_DIR}/{name}_prompt.txt", "w") as fo:
        fo.write((system + "\n\n" if system else "") + prompt)


def manual(name):
    p = f"{MANUAL_DIR}/{name}.txt"
    if os.path.exists(p):
        text = open(p).read().strip()
        return text or None
    return None
