# Day 4.3: 小型 RAG 风控知识库
#
# RAG = 检索增强生成：先从内部文档里检索相关段落，再让大模型只根据这些段落作答
# 解决的问题：大模型不知道公司内部的规则和口径；知识更新只需改文档，不用改 prompt；答案可追溯出处
# 流程：kb/ 下的文档按 "## 小节" 切成片段 -> TF-IDF 向量化（按汉字 2-3 字组合）-> 余弦相似度取前 3
#       -> 相似度太低直接回答"知识库中没有相关信息" -> 否则把片段交给大模型作答并标注出处
# 说明：demo 规模只有二三十个片段，TF-IDF 足够；生产环境一般换成向量模型做语义检索
import glob
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import llm_client

TOP_K = 3
MIN_SIM = 0.12  # demo 问题里，库内问题最高相似度都在 0.2 以上，库外问题约 0.08

chunks = []
for path in sorted(glob.glob("kb/*.md")):
    doc = os.path.basename(path)
    title = ""
    for block in open(path).read().split("\n## "):
        lines = block.strip().splitlines()
        if block.startswith("# "):
            title = lines[0][2:]
            lines = lines[1:]
            if not lines:
                continue
        head, text = lines[0].lstrip("# "), " ".join(lines[1:]).strip()
        if text:
            chunks.append({"doc": doc, "section": f"{title} / {head}", "text": text})

vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
M = vec.fit_transform([c["section"] + " " + c["text"] for c in chunks])


def retrieve(q):
    sims = cosine_similarity(vec.transform([q]), M)[0]
    idx = np.argsort(-sims)[:TOP_K]
    return [(chunks[i], sims[i]) for i in idx]


SYSTEM = ("你是信贷风控团队的内部知识助手。只能根据提供的资料片段回答，并在句末用 [片段编号] 标注出处。"
          "资料中没有的信息，直接回答：知识库中没有相关信息。不得使用资料以外的知识补充内部口径。回答不超过 150 字。")


def answer(q):
    hits = retrieve(q)
    if hits[0][1] < MIN_SIM:
        return hits, "知识库中没有相关信息。（检索相似度过低，未调用大模型）"
    ctx = "\n\n".join(f"[{k + 1}] 《{h['section']}》{h['text']}" for k, (h, _) in enumerate(hits))
    if llm_client.available():
        return hits, llm_client.ask(f"资料片段：\n{ctx}\n\n问题：{q}", system=SYSTEM, max_tokens=500)
    return hits, "（离线模式：未调用大模型，以下为检索到的最相关片段）\n" + hits[0][0]["text"]


DEMO = [
    ("PSI 超过多少要预警？预警之后怎么处理？", "04_monitoring_sop.md"),
    ("为什么表现期不能太短？", "02_label_and_sample.md"),
    ("加验环节假设能拦下多少坏客户？", "03_strategy_rules.md"),
    ("怎么识别多头借贷？", "05_fraud_patterns.md"),
    ("月报里的数字可以让大模型来算吗？", "06_llm_usage_policy.md"),
    ("KS 高于多少要怀疑数据穿越？", "01_model_metrics.md"),
    ("资金成本从 3% 涨到 5%，阈值应该怎么调？", "03_strategy_rules.md"),
    ("美团金融的评分卡用了哪些变量？", None),
]

if len(sys.argv) > 1:
    DEMO = [(" ".join(sys.argv[1:]), None)]

mode = f"LLM ({llm_client.MODEL})" if llm_client.available() else "离线（仅检索）"
print(f"知识库: {len(set(c['doc'] for c in chunks))} 个文档，{len(chunks)} 个片段；作答模式: {mode}\n")
out, hit = [f"# RAG 知识库问答 demo\n\n知识库 {len(chunks)} 个片段，作答模式：{mode}\n"], 0
for q, expect in DEMO:
    hits, ans = answer(q)
    docs = [h["doc"] for h, _ in hits]
    ok = (expect in docs) if expect else (hits[0][1] < MIN_SIM or "没有相关信息" in ans)
    hit += ok
    src = "；".join(f"{h['section']}（相似度 {s:.2f}）" for h, s in hits)
    print(f"Q: {q}\n检索: {src}\nA: {ans}\n{'✓' if ok else '✗'} 期望来源: {expect or '应拒答'}\n")
    out.append(f"## Q: {q}\n\n**检索到的片段**：{src}\n\n**回答**：{ans}\n\n期望来源：{expect or '应拒答'}，"
               f"{'命中' if ok else '未命中'}\n")
print(f"检索命中（前 {TOP_K} 条含期望文档，或正确拒答）: {hit}/{len(DEMO)}")
out.append(f"\n**检索命中率：{hit}/{len(DEMO)}**\n")
if len(sys.argv) == 1:
    open("output/rag_demo.md", "w").write("\n".join(out))
    print("问答记录已存 output/rag_demo.md；也可以自己提问：python scripts/09_rag_kb.py 你的问题")
