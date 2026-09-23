# Day 4.1: 大模型辅助归因 —— 让大模型提假设，用数据逐条校验
#
# 问题：2015 年放款的坏账率比 2014 年高，为什么？
# 流程：代码汇总事实 -> 大模型提出归因假设（每条必须指定一个可检验的变量）-> 代码逐条校验
# 校验方法：把坏账率变化拆成两部分
#   结构效应 = 客群构成变了（例如低 FICO 的人变多了）：Σ (2015 占比 - 2014 占比) × 2014 该组坏账率
#   组内效应 = 同一组人坏账率变了（例如同样 FICO 的人更容易违约了）：Σ 2015 占比 × (2015 坏账率 - 2014 坏账率)
#   如果某条假设说"是因为客群变差"，那它的结构效应必须足够大，否则就是不成立的"幻觉"
import json
import re
import numpy as np
import pandas as pd
import llm_client

f = pd.read_csv("data/features.csv")
o = pd.read_csv("data/loan_outcomes.csv")[["id", "int_rate", "grade"]]
s = pd.read_csv("data/scored.csv")[["id", "score"]]
d = f.merge(o, on="id").merge(s, on="id")
a, b = d[d["issue_year"] == 2014], d[d["issue_year"] == 2015]
gap = b["bad"].mean() - a["bad"].mean()

VARS = ["fico", "dti", "inq_last_6mths", "annual_inc", "loan_amnt", "emp_length_yrs", "revol_util",
        "acc_open_past_24mths", "credit_age_months", "int_rate", "grade", "purpose",
        "home_ownership", "verification_status", "score"]
CAT_VARS = ["grade", "purpose", "home_ownership", "verification_status"]  # 用列表而非集合，保证 prompt 每次顺序相同

facts = {
    "bad_rate_2014": round(a["bad"].mean(), 4), "bad_rate_2015": round(b["bad"].mean(), 4),
    "n_2014": len(a), "n_2015": len(b),
    "mean_by_year": {v: {"2014": round(a[v].mean(), 3), "2015": round(b[v].mean(), 3)}
                     for v in VARS if v not in CAT_VARS},
    "share_by_year": {v: {"2014": a[v].value_counts(normalize=True).round(3).head(6).to_dict(),
                          "2015": b[v].value_counts(normalize=True).round(3).head(6).to_dict()}
                      for v in CAT_VARS},
}

SYSTEM = "你是一名信贷风控策略分析师，擅长对坏账率变化做归因。"
PROMPT = f"""以下是 Lending Club 36 期贷款 2014 年与 2015 年放款客群的汇总数据（JSON）：
{json.dumps(facts, ensure_ascii=False)}

2015 年坏账率比 2014 年高 {gap * 100:.2f} 个百分点。请提出 6 条可能的归因假设。
要求：
1. 每条假设必须对应下面列表中的一个变量，以便用数据检验：{VARS}
2. 说明你认为的作用机制
3. 只输出 JSON 数组，不要任何其他文字，格式：
[{{"hypothesis": "一句话假设", "variable": "变量名", "mechanism": "机制说明"}}]"""

OFFLINE_HYPOTHESES = [
    {"hypothesis": "2015 年申请人信用分下降，客群整体变差", "variable": "fico", "mechanism": "低 FICO 客户占比上升"},
    {"hypothesis": "借款人负债收入比上升，还款压力更大", "variable": "dti", "mechanism": "高 DTI 客户占比上升"},
    {"hypothesis": "多头借贷加剧，近期征信查询增多", "variable": "inq_last_6mths", "mechanism": "查询多的客户更缺钱"},
    {"hypothesis": "平台为扩张放宽准入，低评级贷款占比上升", "variable": "grade", "mechanism": "D-G 级贷款占比上升"},
    {"hypothesis": "收入核验比例下降，虚报收入增多", "variable": "verification_status", "mechanism": "未核验客户占比上升"},
    {"hypothesis": "同等资质客户的违约倾向上升（环境或定价因素）", "variable": "score", "mechanism": "同分数段坏账率上升"},
]

llm_client.save_prompt("attribution", PROMPT, SYSTEM)
if llm_client.available():
    print(f"调用大模型 {llm_client.MODEL} 生成归因假设……")
    raw = llm_client.ask(PROMPT, system=SYSTEM)
    source = f"LLM API ({llm_client.MODEL})"
elif llm_client.manual("attribution"):
    print("读取人工粘贴的 Claude 网页版回答 output/manual_llm/attribution.txt")
    raw = llm_client.manual("attribution")
    source = "Claude 网页版（prompt 与回答人工复制粘贴）"
else:
    print("未检测到大模型回答，使用离线预置假设（仅用于跑通流程，不是大模型的真实输出）")
    print("  -> 把 output/manual_llm/attribution_prompt.txt 的内容发给 Claude 网页版新对话，")
    print("     回答存成 output/manual_llm/attribution.txt 后重跑本脚本")
    raw = None
    source = "offline preset"

if raw is None:
    hyps = OFFLINE_HYPOTHESES
    raw = json.dumps(OFFLINE_HYPOTHESES, ensure_ascii=False, indent=1)
else:
    try:
        hyps = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    except Exception:
        raise SystemExit(f"大模型没有按要求输出 JSON，原始输出：\n{raw}")

with open("output/llm_attribution_prompt_and_raw.txt", "w") as fo:
    fo.write(f"SOURCE: {source}\n\nSYSTEM:\n{SYSTEM}\n\nPROMPT:\n{PROMPT}\n\nRAW OUTPUT:\n{raw}\n")


def decompose(var):
    if var in CAT_VARS:
        ga, gb = a[var].fillna("NA"), b[var].fillna("NA")
    else:
        cuts = np.unique(np.nanquantile(a[var], np.linspace(0.1, 0.9, 9)))
        bins = [-np.inf] + list(cuts) + [np.inf]
        # 组名前加序号，打印明细时能按数值从小到大排
        labels = [f"{i:02d}: {itv}" for i, itv in enumerate(pd.IntervalIndex.from_breaks(bins))]
        ga = pd.cut(a[var], bins, labels=labels).astype(str).where(a[var].notna(), "NA")
        gb = pd.cut(b[var], bins, labels=labels).astype(str).where(b[var].notna(), "NA")
    ta = a.groupby(ga)["bad"].agg(["mean", "size"])
    tb = b.groupby(gb)["bad"].agg(["mean", "size"])
    t = ta.join(tb, lsuffix="_14", rsuffix="_15", how="outer").fillna(0)
    w14, w15 = t["size_14"] / t["size_14"].sum(), t["size_15"] / t["size_15"].sum()
    mix = ((w15 - w14) * t["mean_14"]).sum()
    rate = (w15 * (t["mean_15"] - t["mean_14"])).sum()
    return mix, rate, t


def group_detail(var, t):
    # 组内型假设的证据：同一组客户 2014 -> 2015 坏账率怎么变
    # int_rate 的假设是"定价没跟上风险"，所以改看每个评级的平均利率和坏账率是否同向变化
    if var == "int_rate":
        g = pd.DataFrame({"avg_rate_2014": a.groupby("grade")["int_rate"].mean(),
                          "avg_rate_2015": b.groupby("grade")["int_rate"].mean(),
                          "bad_rate_2014": a.groupby("grade")["bad"].mean(),
                          "bad_rate_2015": b.groupby("grade")["bad"].mean()})
        g[["bad_rate_2014", "bad_rate_2015"]] *= 100
        g.index.name = "grade"
        return "每个 grade 的平均利率(%)与坏账率(%):\n" + g.round(2).to_string()
    g = pd.DataFrame({"bad_rate_2014": t["mean_14"] * 100, "bad_rate_2015": t["mean_15"] * 100})
    g.index.name = var
    return f"按 {var} 分组的坏账率(%):\n" + g.round(2).to_string()


rows = []
for h in hyps:
    v = h.get("variable")
    if v not in VARS:
        rows.append({**h, "mix_effect_pp": np.nan, "within_effect_pp": np.nan, "mix_share": np.nan,
                     "verdict": "无法检验：变量不在数据中"})
        continue
    mix, rate, t = decompose(v)
    share = mix / gap
    # 假设分两类："客群构成变差"看结构效应；"同一组客户风险变高"（评级贬值、定价不足、核验失效等）看组内效应
    claim = h.get("hypothesis", "") + h.get("mechanism", "")
    within_claim = v == "score" or re.search(r"同一|同等|内部|组内|定价|贬值|甄别|未能有效|未被有效", claim) is not None
    detail = group_detail(v, t) if within_claim else ""
    if within_claim:
        within_share = rate / gap
        if within_share >= 0.5:
            verdict = "组内效应为正（该标准区分力弱，结论见分组明细与人工校验记录）"
        elif within_share >= 0.2:
            verdict = "部分成立（按组内效应判定）：有贡献但不是主因"
        else:
            verdict = "不成立（按组内效应判定）：同一组客户坏账率基本没变"
    elif share >= 0.3:
        verdict = "成立：客群结构变化能解释主要部分"
    elif share >= 0.1:
        verdict = "部分成立：有贡献但不是主因"
    elif share > -0.1:
        verdict = "不成立：客群结构几乎没变"
    else:
        verdict = "不成立：方向相反（按这个变量看，客群结构变化反而压低了坏账率）"
    rows.append({**h, "mix_effect_pp": mix * 100, "within_effect_pp": rate * 100,
                 "mix_share": share, "verdict": verdict, "detail": detail})

res = pd.DataFrame(rows)
res.drop(columns="detail").to_csv("output/attribution_check.csv", index=False)

print(f"\n2015 vs 2014 坏账率: {a['bad'].mean():.2%} -> {b['bad'].mean():.2%}（+{gap * 100:.2f} 个百分点）")
print(f"假设来源: {source}\n")
for _, r in res.iterrows():
    print(f"[{r['variable']}] {r['hypothesis']}")
    if pd.notna(r["mix_effect_pp"]):
        print(f"    结构效应 {r['mix_effect_pp']:+.2f}pp（占差距 {r['mix_share']:.0%}），组内效应 {r['within_effect_pp']:+.2f}pp")
    print(f"    结论: {r['verdict']}")
    if isinstance(r.get("detail"), str) and r["detail"]:
        print("    " + r["detail"].replace("\n", "\n    ") + "\n")
ok = res["verdict"].str.startswith("成立").sum()
part = res["verdict"].str.startswith("部分").sum()
pos = res["verdict"].str.startswith("组内效应为正").sum()
print(f"\n共 {len(res)} 条假设：成立 {ok} 条，部分成立 {part} 条，组内效应为正（待人工判断）{pos} 条，"
      f"不成立或无法检验 {len(res) - ok - part - pos} 条")
if pos:
    print("组内型假设的最终结论见 output/attribution_human_review.md")
print("完整 prompt 与模型原始输出已存 output/llm_attribution_prompt_and_raw.txt")
