# Day 2 (2/2): 逻辑回归评分卡 + AUC / KS / PSI 评估 + 树模型对比
#
# 1. 用 WOE 编码后的变量训练逻辑回归（2014 训练，2015 测试）
#    WOE 越大越"好"，所以每个变量的系数都应为负；符号为正或不显著（p > 0.05）的变量逐个剔除
# 2. 把违约概率换算成分数：分数 = offset + factor × ln(好坏比)
#    刻度：好坏比 50:1 时对应 600 分，好坏比每翻一倍加 20 分（PDO = 20）
# 3. 评估：AUC、KS（坏客户与好客户累计分布的最大差距）、PSI（训练期与测试期分数分布的漂移）
# 4. 用同一批原始特征训练梯度提升树，只作对比
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

spec = json.load(open("data/woe_bins.json"))
woe = pd.read_csv("data/woe_features.csv")
tr, te = woe[woe["split"] == "train"], woe[woe["split"] == "test"]


def fit_logit(cols):
    m = LogisticRegression(C=1e9, max_iter=2000).fit(tr[cols], tr["bad"])
    X = np.column_stack([np.ones(len(tr)), tr[cols].values])
    p = m.predict_proba(tr[cols])[:, 1]
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    coef = np.r_[m.intercept_, m.coef_[0]]
    se = np.sqrt(np.diag(cov))
    table = pd.DataFrame({"coef": coef, "se": se, "z": coef / se,
                          "p_value": 2 * norm.sf(np.abs(coef / se))},
                         index=["intercept"] + cols)
    return m, table


cols = list(spec["_selected"])
while True:
    model, coef_table = fit_logit(cols)
    t = coef_table.drop("intercept")
    wrong = t[(t["coef"] > 0) | (t["p_value"] > 0.05)]
    if wrong.empty:
        break
    worst = wrong["p_value"].idxmax() if (wrong["coef"] <= 0).all() else wrong["coef"].idxmax()
    print(f"剔除 {worst}: coef={t.loc[worst, 'coef']:.3f}, p={t.loc[worst, 'p_value']:.3f}")
    cols.remove(worst)

print("\n最终模型系数（系数、标准误、z 值、p 值；标准误由信息矩阵手算）:")
print(coef_table.round(4).to_string())


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return np.max(tpr - fpr)


def to_score(p):
    return OFFSET + FACTOR * np.log((1 - p) / p)


res = {}
for name, d in [("train", tr), ("test", te)]:
    p = model.predict_proba(d[cols])[:, 1]
    res[name] = pd.DataFrame({"id": d["id"].values, "split": name, "bad": d["bad"].values,
                              "pd": p, "score": to_score(p)})
    print(f"{name}: AUC={roc_auc_score(d['bad'], p):.4f}  KS={ks_stat(d['bad'], p):.4f}")

scored = pd.concat(res.values(), ignore_index=True)
scored.to_csv("data/scored.csv", index=False)

# 评分卡表：每个变量每一箱对应多少分
base_points = OFFSET - FACTOR * coef_table.loc["intercept", "coef"]
rows = [{"variable": "BASE", "bin": "", "woe": np.nan, "points": round(base_points)}]
for c in cols:
    b = coef_table.loc[c, "coef"]
    for bin_name, w in spec[c]["woe"].items():
        rows.append({"variable": c, "bin": bin_name, "woe": round(w, 4), "points": round(-FACTOR * b * w)})
card = pd.DataFrame(rows)
card.to_csv("output/scorecard.csv", index=False)
print(f"\n评分卡已存 output/scorecard.csv（基础分 {round(base_points)}，共 {len(card) - 1} 个分箱）")

# PSI：以训练集分数的 10 分位为界，比较测试集落在各箱的比例
edges = np.unique(np.quantile(res["train"]["score"], np.linspace(0, 1, 11)))
edges[0], edges[-1] = -np.inf, np.inf
e = np.histogram(res["train"]["score"], edges)[0] / len(res["train"])
a = np.histogram(res["test"]["score"], edges)[0] / len(res["test"])
psi_parts = (a - e) * np.log(a / e)
psi = psi_parts.sum()
print(f"\n分数 PSI（2014 -> 2015）= {psi:.4f}  （< 0.1 稳定；0.1-0.25 需关注；> 0.25 显著漂移）")

# 分数十等分坏账率：检验分数越低坏账率越高
res["test"]["decile"] = pd.qcut(res["test"]["score"], 10, labels=False) + 1
dec = res["test"].groupby("decile").agg(score_min=("score", "min"), score_max=("score", "max"),
                                        n=("bad", "size"), bad_rate=("bad", "mean"))
print("\n测试集按分数十等分的坏账率（1 = 最低分）:")
print(dec.round(3).to_string())

# 树模型对比：用原始特征（不做 WOE），能自己处理缺失值
raw = pd.read_csv("data/features.csv")
num_cols = [c for c in raw.columns if c not in ["id", "issue_d", "issue_year", "split", "bad",
                                                "home_ownership", "verification_status", "purpose"]]
X = raw[num_cols].copy()
for c in ["home_ownership", "verification_status", "purpose"]:
    X[c] = raw[c].astype("category").cat.codes
is_tr = raw["split"] == "train"
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
gb.fit(X[is_tr], raw.loc[is_tr, "bad"])
p_gb = gb.predict_proba(X[~is_tr])[:, 1]
auc_gb, ks_gb = roc_auc_score(raw.loc[~is_tr, "bad"], p_gb), ks_stat(raw.loc[~is_tr, "bad"], p_gb)
auc_lr = roc_auc_score(res["test"]["bad"], res["test"]["pd"])
ks_lr = ks_stat(res["test"]["bad"], res["test"]["pd"])
print(f"\n测试集对比  评分卡: AUC={auc_lr:.4f} KS={ks_lr:.4f}   梯度提升树: AUC={auc_gb:.4f} KS={ks_gb:.4f}")

with open("output/model_metrics.txt", "w") as f:
    f.write(f"variables: {cols}\n")
    for n in ["train", "test"]:
        f.write(f"{n}: AUC={roc_auc_score(res[n]['bad'], res[n]['pd']):.4f} KS={ks_stat(res[n]['bad'], res[n]['pd']):.4f}\n")
    f.write(f"score PSI train->test: {psi:.4f}\n")
    f.write(f"GBDT test: AUC={auc_gb:.4f} KS={ks_gb:.4f}\n")

# 图 1：ROC 与 KS 曲线（测试集）
yt, pt = res["test"]["bad"], res["test"]["pd"]
fpr, tpr, thr = roc_curve(yt, pt)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(fpr, tpr, label=f"Scorecard AUC={auc_lr:.3f}")
fg, tg, _ = roc_curve(raw.loc[~is_tr, "bad"], p_gb)
ax[0].plot(fg, tg, label=f"GBDT AUC={auc_gb:.3f}", alpha=0.7)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (2015 out-of-time)")
ax[0].legend()
ss = res["test"].sort_values("score")
cum_bad = ss["bad"].cumsum() / ss["bad"].sum()
cum_good = (1 - ss["bad"]).cumsum() / (1 - ss["bad"]).sum()
k = np.argmax((cum_bad - cum_good).values)
ax[1].plot(ss["score"], cum_bad, label="Cumulative bad")
ax[1].plot(ss["score"], cum_good, label="Cumulative good")
ax[1].vlines(ss["score"].iloc[k], cum_good.iloc[k], cum_bad.iloc[k], colors="r",
             label=f"KS={cum_bad.iloc[k] - cum_good.iloc[k]:.3f} at score {ss['score'].iloc[k]:.0f}")
ax[1].set(xlabel="Score", ylabel="Cumulative share", title="KS curve (2015 out-of-time)")
ax[1].legend()
plt.tight_layout()
plt.savefig("output/fig1_roc_ks.png", dpi=150)

# 图 2：分数分布与分数段坏账率
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
s = res["test"]
ax[0].hist(s.loc[s["bad"] == 0, "score"], bins=50, alpha=0.6, density=True, label="Good")
ax[0].hist(s.loc[s["bad"] == 1, "score"], bins=50, alpha=0.6, density=True, label="Bad")
ax[0].set(xlabel="Score", title="Score distribution by outcome (2015)")
ax[0].legend()
ax[1].bar(dec.index, dec["bad_rate"] * 100)
ax[1].set(xlabel="Score decile (1 = lowest score)", ylabel="Bad rate (%)", title="Bad rate by score decile (2015)")
plt.tight_layout()
plt.savefig("output/fig2_score_badrate.png", dpi=150)
print("\n图已存 output/fig1_roc_ks.png, output/fig2_score_badrate.png")
