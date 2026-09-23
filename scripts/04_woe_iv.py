# Day 2 (1/2): WOE 分箱与 IV 筛选
#
# WOE（证据权重）= ln(该箱好客户占全部好客户的比例 / 该箱坏客户占全部坏客户的比例)
#   > 0 说明这一箱好人偏多，< 0 说明坏人偏多
# IV（信息值）= Σ (好客户占比 - 坏客户占比) × WOE，衡量一个变量整体能把好坏分开多少
#   经验门槛：< 0.02 没用；0.02-0.1 弱；0.1-0.3 中等；> 0.3 强
#
# 分箱规则：
#   数值变量先按训练集 10 分位切箱，再把坏账率不单调的相邻箱合并，直到单调
#   （单调是评分卡的业务要求：比如查询次数越多，扣分只能越多，不能忽高忽低）
#   缺失值单独成一箱；类别变量每个类别一箱，占比 < 1% 的并入 OTHER
# 所有分箱和 WOE 只用训练集（2014）计算，再套用到测试集，避免测试集信息泄漏
import json
import numpy as np
import pandas as pd

NUM = [
    "loan_amnt", "annual_inc", "dti", "emp_length_yrs", "fico", "credit_age_months",
    "inq_last_6mths", "delinq_2yrs", "ever_delinq", "open_acc", "total_acc", "pub_rec",
    "pub_rec_bankruptcies", "revol_bal", "revol_util", "mort_acc", "acc_open_past_24mths",
    "bc_util", "percent_bc_gt_75", "num_tl_op_past_12m", "loan_to_income", "installment_to_income",
]
CAT = ["home_ownership", "verification_status", "purpose"]
IV_MIN = 0.02
CORR_MAX = 0.7

df = pd.read_csv("data/features.csv")
train = df[df["split"] == "train"]
y = train["bad"]
n_bad, n_good = y.sum(), (1 - y).sum()


def woe_table(bin_labels, y):
    t = pd.DataFrame({"bin": bin_labels.values, "bad": y.values})
    g = t.groupby("bin", sort=True)["bad"].agg(n="count", bad="sum")
    g["good"] = g["n"] - g["bad"]
    g["bad_rate"] = g["bad"] / g["n"]
    g["good_pct"] = (g["good"] + 0.5) / n_good
    g["bad_pct"] = (g["bad"] + 0.5) / n_bad
    g["woe"] = np.log(g["good_pct"] / g["bad_pct"])
    g["iv_part"] = (g["good_pct"] - g["bad_pct"]) * g["woe"]
    return g


def cut_num(x, cuts):
    bins = [-np.inf] + list(cuts) + [np.inf]
    labels = [f"{i:02d}: {itv}" for i, itv in enumerate(pd.IntervalIndex.from_breaks(bins))]
    lab = pd.cut(x, bins, labels=labels).astype(str)
    return lab.where(x.notna(), "MISSING")


def monotonic_cuts(x, y):
    cuts = sorted(set(np.nanquantile(x, np.linspace(0.1, 0.9, 9))))
    ok = x.notna()
    xs, ys = x[ok], y[ok]
    direction = np.sign(np.corrcoef(xs.rank(), ys)[0, 1]) or 1
    while cuts:
        idx = np.searchsorted(cuts, xs, side="left")
        rates = ys.groupby(idx).mean().sort_index().values
        diffs = np.diff(rates) * direction
        bad_pos = np.where(diffs < 0)[0]
        if len(bad_pos) == 0:
            break
        cuts.pop(bad_pos[0])
    return cuts


spec, tables, iv = {}, [], {}
for col in NUM:
    cuts = monotonic_cuts(train[col], y)
    g = woe_table(cut_num(train[col], cuts), y)
    spec[col] = {"type": "num", "cuts": cuts, "woe": g["woe"].to_dict()}
    iv[col] = g["iv_part"].sum()
    tables.append(g.assign(var=col))

for col in CAT:
    share = train[col].value_counts(normalize=True)
    keep = share[share >= 0.01].index.tolist()
    lab = train[col].where(train[col].isin(keep), "OTHER").fillna("OTHER")
    g = woe_table(lab, y)
    spec[col] = {"type": "cat", "keep": keep, "woe": g["woe"].to_dict()}
    iv[col] = g["iv_part"].sum()
    tables.append(g.assign(var=col))


def to_woe(data, col):
    s = spec[col]
    if s["type"] == "num":
        lab = cut_num(data[col], s["cuts"])
    else:
        lab = data[col].where(data[col].isin(s["keep"]), "OTHER").fillna("OTHER")
    return lab.map(s["woe"]).fillna(0.0)


woe = pd.DataFrame({c: to_woe(df, c) for c in NUM + CAT})
woe[["id", "split", "bad"]] = df[["id", "split", "bad"]]
woe.to_csv("data/woe_features.csv", index=False)

iv_s = pd.Series(iv).sort_values(ascending=False)

# 筛选：先按 IV 门槛，再剔除高度相关的变量（两两相关 > 0.7 时保留 IV 高的）
cand = iv_s[iv_s >= IV_MIN].index.tolist()
corr = woe.loc[woe["split"] == "train", cand].corr().abs()
selected = []
for c in cand:
    if all(corr.loc[c, s] <= CORR_MAX for s in selected):
        selected.append(c)
dropped_corr = [c for c in cand if c not in selected]

spec["_selected"] = selected
with open("data/woe_bins.json", "w") as f:
    json.dump(spec, f, indent=1, default=str)

pd.concat(tables).reset_index().to_csv("output/woe_bins_table.csv", index=False)
iv_out = pd.DataFrame({"iv": iv_s.round(4)})
iv_out["status"] = np.where(iv_out["iv"] < IV_MIN, "drop: low IV",
                   np.where(iv_out.index.isin(dropped_corr), "drop: correlated", "keep"))
iv_out.to_csv("output/iv_table.csv")

print("IV 排名:")
print(iv_out.to_string())
print(f"\n入模变量 {len(selected)} 个: {selected}")
print(f"因高度相关剔除: {dropped_corr}")

print("\n手算示例：inq_last_6mths（近 6 个月征信查询次数）")
ex = pd.concat(tables)
ex = ex[ex["var"] == "inq_last_6mths"][["n", "bad", "good", "bad_rate", "good_pct", "bad_pct", "woe", "iv_part"]]
print(ex.round(4).to_string())
print(f"IV 合计 = {ex['iv_part'].sum():.4f}")
