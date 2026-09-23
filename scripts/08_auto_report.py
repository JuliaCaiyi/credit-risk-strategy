# Day 4.2: 自动化风险策略月报
#
# 设计原则：数字全部由代码计算，大模型只负责组织语言和写解读
#   原因：让大模型自己算数会产生幻觉，风控报告里一个错数字就可能导致错误决策
# 护栏：报告生成后，自动把大模型写出的每个数字和代码提供的事实逐一比对，对不上的标红提示
# Prompt 迭代：v1 只给数据让它写报告；v2 明确"只能引用给定数字、不许推算、不确定就不写"
# 两版都跑一遍，对比未核实数字的个数，记录在 output/prompt_iteration.md
import json
import re
import numpy as np
import pandas as pd
import llm_client

mon = pd.read_csv("output/monthly_monitoring.csv")
tiers = pd.read_csv("output/tier_strategy.csv")
cur, prev = mon.iloc[-1], mon.iloc[-2]
hist = mon.iloc[:-1]

def md_table(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# ---------- 1. 代码计算全部数字 ----------
kpi = []
for col, name in [("approve_rate", "通过率"), ("bad_rate_approved", "通过客群坏账率"),
                  ("psi_score", "分数 PSI"), ("psi_fico", "FICO PSI"), ("psi_dti", "DTI PSI"),
                  ("psi_inq", "征信查询 PSI")]:
    mu, sd = hist[col].mean(), hist[col].std()
    z = (cur[col] - mu) / sd if sd > 0 else 0
    pct = col in ("approve_rate", "bad_rate_approved")
    kpi.append({"指标": name, "本月": round(cur[col] * 100, 2) if pct else round(cur[col], 4),
                "上月": round(prev[col] * 100, 2) if pct else round(prev[col], 4),
                "环比变化": round((cur[col] - prev[col]) * 100, 2) if pct else round(cur[col] - prev[col], 4),
                "前 11 个月均值": round(mu * 100, 2) if pct else round(mu, 4),
                "偏离度(z)": round(z, 2), "异常": "是" if abs(z) >= 2 else "否",
                "单位": "%" if pct else ""})
kpi = pd.DataFrame(kpi)
anomalies = kpi[kpi["异常"] == "是"]["指标"].tolist()
facts = {"报告月份": cur["month"], "放款笔数": int(cur["n"]), "预警状态": cur["alert"],
         "指标": kpi.drop(columns=["异常"]).to_dict(orient="records"),
         "异常指标（|z|>=2）": anomalies,
         "判定规则": {"PSI 黄色预警线": 0.1, "PSI 红色预警线": 0.25, "异常判定": "偏离度 |z| >= 2"},
         "分层策略": tiers.assign(share=(tiers["share"] * 100).round(1),
                              bad_rate_before_action=(tiers["bad_rate_before_action"] * 100).round(2)
                              ).to_dict(orient="records")}

PROMPT_V1 = f"""根据以下风控监控数据，写一份风险策略月报的解读部分和建议动作。
数据：{json.dumps(facts, ensure_ascii=False)}"""

PROMPT_V2 = f"""你在为信贷风控团队撰写月报的"解读与建议"部分。以下 JSON 是代码计算好的全部事实：
{json.dumps(facts, ensure_ascii=False)}

写作规则（必须遵守）：
1. 只能引用 JSON 里出现的数字，原样引用，不得自行计算新的数字（包括差值、比例、倍数）
2. 没有数据支撑的判断不写；推测性表述必须用"可能""建议核查"等措辞
3. 结构：【本月概况】2-3 句；【异常与关注点】逐条对应"异常指标"，没有异常就写"本月无指标触发异常阈值"；【建议动作】不超过 3 条，每条说明依据哪个指标
4. 不做授信决策建议（如调整某个客户的额度），只做策略层面的监控建议
5. 总长度不超过 300 字"""


def template_text():
    lines = [f"【本月概况】{cur['month']} 放款 {int(cur['n'])} 笔，按当前阈值通过率 "
             f"{cur['approve_rate'] * 100:.2f}%，通过客群坏账率 {cur['bad_rate_approved'] * 100:.2f}%，"
             f"整体预警状态 {cur['alert']}。"]
    lines.append("【异常与关注点】" + ("本月无指标触发异常阈值。" if not anomalies else
                 "；".join(f"{a} 偏离历史均值超过 2 个标准差" for a in anomalies) + "。"))
    lines.append("【建议动作】1. 维持现有阈值，继续按月跟踪分数段坏账率；2. 各 PSI 均低于 0.1，暂不需要模型重训。")
    return "\n".join(lines)


def check_numbers(text):
    allowed = set()
    for v in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(facts, ensure_ascii=False)):
        x = float(v)
        for y in (x, abs(x)):  # 文字里常写"下降 0.73"，不带负号
            allowed |= {round(y, 4), round(y, 2), round(y, 1), round(y)}
    text = re.sub(r"\d{4}-\d{2}|\d{4}年\d{1,2}月", "", text)  # 去掉月份
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 去掉千位分隔符：30,241 -> 30241
    text = re.sub(r"(?m)^\s*\d+[.、)]\s", "", text)  # 去掉行首的列表序号
    # 注意不能用 \w：Python 里汉字也算 \w，会漏掉紧挨汉字的数字（如"近0.6个百分点"）
    found = re.findall(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?", text)
    bad = []
    for v in found:
        x = float(v)
        if re.fullmatch(r"20\d\d", v):
            continue  # 年份
        if not any(abs(x - y) < 1e-9 for y in {round(x, 4), round(x, 2), round(x, 1), round(x)} & allowed):
            bad.append(v)
    return bad


llm_client.save_prompt("report_v1", PROMPT_V1)
llm_client.save_prompt("report_v2", PROMPT_V2)
out = {}
if llm_client.available():
    for ver, prompt in [("v1", PROMPT_V1), ("v2", PROMPT_V2)]:
        print(f"调用大模型生成解读（prompt {ver}）……")
        text = llm_client.ask(prompt)
        out[ver] = (text, check_numbers(text))
    src_name = f"LLM API ({llm_client.MODEL})"
else:
    for ver in ["v1", "v2"]:
        text = llm_client.manual(f"report_{ver}")
        if text:
            out[ver] = (text, check_numbers(text))
    src_name = "Claude 网页版（人工复制粘贴）"

if out:
    with open("output/prompt_iteration.md", "w") as fo:
        fo.write(f"# Prompt 迭代记录（{src_name}）\n\n")
        for ver, prompt in [("v1", PROMPT_V1), ("v2", PROMPT_V2)]:
            if ver in out:
                text, unv = out[ver]
                fo.write(f"## {ver}\n\n### Prompt\n```\n{prompt}\n```\n\n### 输出\n{text}\n\n"
                         f"### 数字核对：{len(unv)} 个数字不在事实中 {unv}\n\n")
    for ver in out:
        print(f"prompt {ver}：未核实数字 {len(out[ver][1])} 个 {out[ver][1]}")
    print("详见 output/prompt_iteration.md")
if "v2" in out:
    body, unverified = out["v2"]
    source = f"{src_name}，prompt v2"
else:
    print("未检测到大模型回答，解读部分使用模板生成")
    print("  -> 把 output/manual_llm/report_v1_prompt.txt 和 report_v2_prompt.txt 分别发给 Claude 网页版新对话，")
    print("     回答存成 output/manual_llm/report_v1.txt、report_v2.txt 后重跑本脚本")
    body = template_text()
    unverified = check_numbers(body)
    source = "template (offline)"

md = [f"# 风险策略月报 · {cur['month']}", "",
      f"> 数字由代码计算；解读由 {source} 生成，并经自动数字核对。", "",
      "## 一、核心指标（代码计算）", "", md_table(kpi), "",
      "## 二、分层策略运行情况", "",
      tiers.assign(share=(tiers["share"] * 100).round(1),
                   bad_rate_before_action=(tiers["bad_rate_before_action"] * 100).round(2)
                   ).rename(columns={"share": "占比%", "bad_rate_before_action": "处置前坏账率%"}).pipe(md_table),
      "", "## 三、解读与建议", "", body, "",
      "## 四、数字核对", "",
      "解读中的所有数字均可在上方事实中找到。" if not unverified else
      f"**警告：以下数字未在事实数据中找到，需人工核实：{unverified}**", "",
      "## 五、口径说明", "",
      "坏账率为贷款最终结局回看口径。实际线上监控时，近期放款尚未到期，需改用首期逾期率、30 天以上逾期率等早期指标。"]
with open(f"output/report_{cur['month']}.md", "w") as fo:
    fo.write("\n".join(md))

print("\n".join(md))
