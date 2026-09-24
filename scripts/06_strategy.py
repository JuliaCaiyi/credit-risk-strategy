# Day 3: 策略层 —— 阈值优化、三层分层策略、策略评估体系（v2）
#
# v2 修订：所有阈值只在 2014 年（选择集）上确定，再原样用到 2015 年（评估集）算收益
#   v1 在 2015 年上选阈值又在 2015 年上算收益，相当于先看答案再定策略，结果偏乐观
#   2015 年自身的最优解保留为"事后上限"，它和样本外结果的差距就是阈值漂移的代价
#
# 收益口径（每笔贷款，用真实还款现金流回测）：
#   净收益 = 实际收回总额 total_pymnt - 放款本金 - 资金与运营成本
#   资金与运营成本 = 本金 × 年化成本率 K × 平均占用年限 1.5（36 期等额本息，平均未还本金约为一半）
#   被拒绝：收益为 0
# 注意：这里的"通过率"是在 LC 已放款客群中的通过率，LC 拒掉的申请人不在数据里
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_BASE = 0.03
YEARS = 1.5
CATCH_BAD = 0.40    # 加验能拦下的坏客户比例（假设）
DROP_GOOD = 0.10    # 加验导致好客户流失的比例（假设）
VERIFY_COST = 30    # 每笔加验成本，美元（假设）
V_MAX = 0.20        # 加验比例上限
P_MIN = 0.85        # 最终通过率下限
SWAP_RATE = 0.85    # Swap Set 比较时统一的通过率

s = pd.read_csv("data/scored.csv")
o = pd.read_csv("data/loan_outcomes.csv")[["id", "loan_amnt", "total_pymnt", "int_rate"]]
f = pd.read_csv("data/features.csv")[["id", "issue_d", "fico", "dti", "inq_last_6mths"]]
d = s.merge(o, on="id").merge(f, on="id")


def pnl(data, k=K_BASE):
    return data["total_pymnt"] - data["loan_amnt"] * (1 + k * YEARS)


d["pnl"] = pnl(d)
tr = d[d["split"] == "train"].sort_values("score", ascending=False).reset_index(drop=True)
te = d[d["split"] == "test"].sort_values("score", ascending=False).reset_index(drop=True)


def curve(data, k=K_BASE):
    """按分数从高到低依次放行：每个位置的通过率、坏账率、累计净收益"""
    n = len(data)
    cp = pnl(data, k).cumsum().values
    return np.arange(1, n + 1) / n, data["bad"].cumsum().values / np.arange(1, n + 1), cp


def evaluate(data, cut):
    ok = data["score"] >= cut
    return {"approve": ok.mean(), "bad": data.loc[ok, "bad"].mean() if ok.any() else np.nan,
            "profit": data.loc[ok, "pnl"].sum()}


# ---------- 3.1 单阈值：在 2014 上选，在 2015 上评估 ----------
a_tr, b_tr, c_tr = curve(tr)
cut_opt = tr["score"][c_tr.argmax()]
cut_int = tr["score"][np.where(b_tr <= 0.10)[0].max()]
a_te, b_te, c_te = curve(te)
cut_hind = te["score"][c_te.argmax()]

rows = []
for name, cut in [("全部通过", -np.inf), ("风控直觉：2014 年坏账率 <= 10% 的分数线", cut_int),
                  ("收益最优：2014 年选出的分数线", cut_opt), ("事后上限：2015 年自身最优（不可实现）", cut_hind)]:
    r = evaluate(te, cut)
    rows.append({"方案": name, "分数线": round(cut, 1) if np.isfinite(cut) else "-",
                 "2015 通过率": r["approve"], "2015 坏账率": r["bad"], "2015 净收益 $M": r["profit"] / 1e6})
single = pd.DataFrame(rows)
print("== 3.1 单阈值（2014 选阈值 -> 2015 样本外评估，年化成本 3%）==")
print(single.round(4).to_string(index=False))
single.to_csv("output/single_cutoff_oos.csv", index=False)
oos = evaluate(te, cut_opt)
hind = evaluate(te, cut_hind)
intu = evaluate(te, cut_int)
def more_less(diff):
    return f"{'多赚' if diff >= 0 else '少赚'} ${abs(diff) / 1e6:.1f}M"


print(f"\n样本外：收益最优分数线比直觉分数线{more_less(oos['profit'] - intu['profit'])}，"
      f"比全部通过{more_less(oos['profit'] - te['pnl'].sum())}")
print(f"阈值漂移代价：事后上限 - 样本外 = ${(hind['profit'] - oos['profit']) / 1e6:.1f}M")

# 边际分析：在选阈值的同一份数据（2014）上、以 2.5 个百分点为一档，避免 1% 小切片的噪音
print("\n边际分析（2014 选择集）：通过率每多放开 2.5 个百分点")
marg = []
n = len(tr)
for a in np.arange(0.70, 0.976, 0.025):
    i0, i1 = int(n * a), int(n * (a + 0.025))
    sl = tr.iloc[i0:i1]
    marg.append({"通过率区间": f"{a:.1%} -> {a + 0.025:.1%}", "新增客群坏账率": sl["bad"].mean(),
                 "净收益变化 $K": sl["pnl"].sum() / 1e3})
marg = pd.DataFrame(marg)
print(marg.round(3).to_string(index=False))
print(f"2014 收益最优通过率 {a_tr[c_tr.argmax()]:.1%}（分数线 {cut_opt:.1f}）：净收益变化由正转负的位置应与此一致")
marg.to_csv("output/marginal_approval.csv", index=False)

print("\n成本敏感性（2014 选阈值 -> 2015 评估）:")
sens = []
for k in [0.0, 0.02, 0.03, 0.04, 0.05]:
    a_k, _, c_k = curve(tr, k)
    cut_k = tr["score"][c_k.argmax()]
    ok = te["score"] >= cut_k
    sens.append({"年化成本 K": k, "2014 选出的通过率": a_k[c_k.argmax()], "分数线": cut_k,
                 "2015 通过率": ok.mean(), "2015 净收益 $M": pnl(te[ok], k).sum() / 1e6})
sens = pd.DataFrame(sens)
print(sens.round(4).to_string(index=False))
sens.to_csv("output/cost_sensitivity.csv", index=False)

# 图 3：两年的收益曲线（按每 1000 笔申请折算，便于比较）+ 两条分数线
fig, ax = plt.subplots(figsize=(9, 5))
step = slice(None, None, 100)
ax.plot(tr["score"][step], c_tr[step] / len(tr) * 1000 / 1e3, label="2014 (selection set)")
ax.plot(te["score"][step], c_te[step] / len(te) * 1000 / 1e3, label="2015 (evaluation set)")
ax.axvline(cut_opt, color="C0", ls="--", lw=1, label=f"2014-optimal cutoff {cut_opt:.0f}")
ax.axvline(cut_hind, color="C1", ls=":", lw=1, label=f"2015 hindsight optimum {cut_hind:.0f}")
ax.set(xlabel="Score cutoff (approve if score >= cutoff)", ylabel="Net profit per 1,000 loans ($K)",
       title=f"Cutoff chosen on 2014, applied to 2015 (cost {K_BASE:.0%}/yr)")
ax.legend()
plt.tight_layout()
plt.savefig("output/fig3_cutoff_tradeoff.png", dpi=150)


# ---------- 3.2 三层分层策略：在 2014 上网格搜索两个阈值，再用到 2015 ----------
def tier_grid(data, catch=CATCH_BAD, drop=DROP_GOOD, cost=VERIFY_COST, p_min=P_MIN):
    asc = data.sort_values("score").reset_index(drop=True)
    N = len(asc)
    grid = np.unique(np.r_[0, (np.linspace(0, 1, 201) * N).astype(int), N])
    b = asc["bad"].values
    cs = lambda v: np.r_[0, np.cumsum(v)][grid]
    Cn = grid.astype(float)
    Cgp, Cbp = cs(np.where(b == 0, asc["pnl"], 0)), cs(np.where(b == 1, asc["pnl"], 0))
    Cg, Cb = cs(1 - b), cs(b)
    I, J = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
    mid_n = Cn[J] - Cn[I]
    prof = ((Cgp[J] - Cgp[I]) * (1 - drop) + (Cbp[J] - Cbp[I]) * (1 - catch) - cost * mid_n
            + (Cgp[-1] + Cbp[-1] - Cgp[J] - Cbp[J]))
    appr = ((N - Cn[J]) + (Cg[J] - Cg[I]) * (1 - drop) + (Cb[J] - Cb[I]) * (1 - catch)) / N
    ok = (I <= J) & (appr >= p_min) & (mid_n / N <= V_MAX)
    i, j = np.unravel_index(np.where(ok, prof, -np.inf).argmax(), prof.shape)
    t = lambda g: asc["score"].iloc[min(grid[g], N - 1)] if grid[g] < N else np.inf
    return (t(i) if grid[i] > 0 else -np.inf), t(j)


def tier_eval(data, t1, t2, catch=CATCH_BAD, drop=DROP_GOOD, cost=VERIFY_COST):
    rej, mid, top = data["score"] < t1, (data["score"] >= t1) & (data["score"] < t2), data["score"] >= t2
    m = data[mid]
    g, b = m[m["bad"] == 0], m[m["bad"] == 1]
    profit = (data.loc[top, "pnl"].sum() + g["pnl"].sum() * (1 - drop) + b["pnl"].sum() * (1 - catch)
              - cost * len(m))
    approved = top.sum() + len(g) * (1 - drop) + len(b) * (1 - catch)
    bad_n = data.loc[top, "bad"].sum() + len(b) * (1 - catch)
    return {"profit": profit, "approve": approved / len(data), "bad": bad_n / approved,
            "reject_share": rej.mean(), "verify_share": mid.mean(), "top_share": top.mean(),
            "mid_good_pnl": g["pnl"].sum(), "mid_bad_pnl": b["pnl"].sum(), "mid_n": len(m)}


t1, t2 = tier_grid(tr)
tier = tier_eval(te, t1, t2)
lift = tier["profit"] / oos["profit"] - 1
print(f"\n== 3.2 三层策略（阈值在 2014 上求解：通过率 >= {P_MIN:.0%}，加验比例 <= {V_MAX:.0%}）==")
print(f"阈值：< {t1:.1f} 拒绝；{t1:.1f}-{t2:.1f} 加验；>= {t2:.1f} 直接通过")
print(f"2015 样本外：拒绝 {tier['reject_share']:.1%} / 加验 {tier['verify_share']:.1%} / 直接通过 {tier['top_share']:.1%}；"
      f"最终通过率 {tier['approve']:.1%}，坏账率 {tier['bad']:.2%}，净收益 ${tier['profit'] / 1e6:.1f}M")
print(f"相对单阈值（样本外 ${oos['profit'] / 1e6:.1f}M）提升 {lift:+.1%}（基于加验假设：拦截 {CATCH_BAD:.0%} 坏客户、"
      f"流失 {DROP_GOOD:.0%} 好客户、每笔 ${VERIFY_COST}）")

# 盈亏平衡：加验层里，拦截率至少多高，加验才比"直接放行这一层"划算
G, B, nm = tier["mid_good_pnl"], tier["mid_bad_pnl"], tier["mid_n"]
be = -(DROP_GOOD * G + VERIFY_COST * nm) / B
print(f"盈亏平衡：加验至少要拦下加验层中 {be:.1%} 的坏客户，才比直接放行这一层划算"
      f"（好客户流失 {DROP_GOOD:.0%}、每笔 ${VERIFY_COST} 不变）")
be_rows = []
for dg in [0.05, 0.10, 0.20]:
    for vc in [10, 30, 60]:
        be_rows.append({"好客户流失率": dg, "每笔加验成本 $": vc, "盈亏平衡拦截率": -(dg * G + vc * nm) / B})
be_tab = pd.DataFrame(be_rows)
print(be_tab.round(3).to_string(index=False))
be_tab.to_csv("output/verify_breakeven.csv", index=False)

layers = []
for name, lo, hi, action in [("L1 Approve", t2, np.inf, "直接通过"), ("L2 Verify", t1, t2, "加验后放款"),
                             ("L3 Reject", -np.inf, t1, "拒绝")]:
    seg = te[(te["score"] >= lo) & (te["score"] < hi)]
    layers.append({"layer": name, "score_range": f"[{lo:.0f}, {hi:.0f})", "share": len(seg) / len(te),
                   "bad_rate_before_action": seg["bad"].mean(), "action": action})
layers = pd.DataFrame(layers)
print("\n三层分层策略表（2015 样本外）:")
print(layers.round(4).to_string(index=False))
layers.to_csv("output/tier_strategy.csv", index=False)

print("\n加验假设敏感性（每组假设都在 2014 上重新求阈值，再到 2015 评估）:")
sv = []
for cb in [0.2, 0.3, 0.4]:
    for dg in [0.05, 0.10, 0.20]:
        a1, a2 = tier_grid(tr, catch=cb, drop=dg)
        r = tier_eval(te, a1, a2, catch=cb, drop=dg)
        sv.append({"拦截率": cb, "好客户流失率": dg, "2015 净收益 $M": r["profit"] / 1e6,
                   "相对单阈值": r["profit"] / oos["profit"] - 1})
sv = pd.DataFrame(sv)
print(sv.round(3).to_string(index=False))
print(f"提升区间：{sv['相对单阈值'].min():+.1%} 到 {sv['相对单阈值'].max():+.1%}")
sv.to_csv("output/verify_sensitivity.csv", index=False)

# 风险-收益前沿（2014 选择集）：不同通过率下限对应的最优单阈值与三层策略
front = []
for p in np.arange(0.70, 0.976, 0.025):
    j = int(len(tr) * p) - 1
    a1, a2 = tier_grid(tr, p_min=p)
    r3 = tier_eval(tr, a1, a2)
    front.append({"approve": p, "single_profit_$M": c_tr[j] / 1e6, "single_bad_rate": b_tr[j],
                  "tier_profit_$M": r3["profit"] / 1e6, "tier_bad_rate": r3["bad"]})
front = pd.DataFrame(front)
front.to_csv("output/risk_return_frontier.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(front["single_bad_rate"] * 100, front["single_profit_$M"], "o-", label="Single cutoff")
ax.plot(front["tier_bad_rate"] * 100, front["tier_profit_$M"], "s-", label="3-tier (min approval as labeled)")
for _, r in front.iterrows():
    ax.annotate(f"{r['approve']:.0%}", (r["single_bad_rate"] * 100, r["single_profit_$M"]),
                textcoords="offset points", xytext=(4, -10), fontsize=7)
ax.set(xlabel="Bad rate of approved (%)", ylabel="Net profit ($M)", title="Risk-return frontier (2014 selection set)")
ax.legend()
plt.tight_layout()
plt.savefig("output/fig4_risk_return_frontier.png", dpi=150)

# ---------- 3.3 策略评估 ----------
print("\n== 3.3 策略评估 ==")
new_ok = te["score"] >= te["score"].quantile(1 - SWAP_RATE)
old_ok = te["fico"] >= te["fico"].quantile(1 - SWAP_RATE)
groups = {"都通过": new_ok & old_ok, "Swap in（旧拒新过）": new_ok & ~old_ok,
          "Swap out（旧过新拒）": ~new_ok & old_ok, "都拒绝": ~new_ok & ~old_ok}
swap = pd.DataFrame([{"group": g, "n": m.sum(), "share": m.mean(), "bad_rate": te.loc[m, "bad"].mean()}
                     for g, m in groups.items()])
print(f"Swap Set（2015；旧：只看 FICO；新：评分卡；通过率都设为 {SWAP_RATE:.0%}）:")
print(swap.round(4).to_string(index=False))
print(f"旧策略坏账率 {te.loc[old_ok, 'bad'].mean():.2%} -> 新策略 {te.loc[new_ok, 'bad'].mean():.2%}")
swap.to_csv("output/swap_set.csv", index=False)


def psi(base, cur, bins=10):
    edges = np.unique(np.nanquantile(base, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(base[~np.isnan(base)], edges)[0] / np.sum(~np.isnan(base))
    a = np.histogram(cur[~np.isnan(cur)], edges)[0] / np.sum(~np.isnan(cur))
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return np.sum((a - e) * np.log(a / e))


te["month"] = pd.to_datetime(te["issue_d"], format="%b-%Y")
mon = []
for m, g in te.groupby("month"):
    ok = g["score"] >= cut_opt
    mon.append({"month": m.strftime("%Y-%m"), "n": len(g), "approve_rate": ok.mean(),
                "bad_rate_approved": g.loc[ok, "bad"].mean(),
                "psi_score": psi(tr["score"].values, g["score"].values),
                "psi_fico": psi(tr["fico"].values, g["fico"].values),
                "psi_dti": psi(tr["dti"].values, g["dti"].values),
                "psi_inq": psi(tr["inq_last_6mths"].values.astype(float), g["inq_last_6mths"].values.astype(float))})
mon = pd.DataFrame(mon)
psi_cols = [c for c in mon.columns if c.startswith("psi")]
mon["alert"] = np.where(mon[psi_cols].max(axis=1) > 0.25, "RED",
               np.where(mon[psi_cols].max(axis=1) > 0.1, "AMBER", "OK"))
print("\n2015 月度监控（以 2014 为基准；PSI > 0.1 黄色预警，> 0.25 红色预警）:")
print(mon.round(4).to_string(index=False))
mon.to_csv("output/monthly_monitoring.csv", index=False)

# 校准：模型预测的违约率准不准（KS 只看排序，不看概率高低）
cal = []
edges = np.unique(np.quantile(tr["score"], np.linspace(0, 1, 11)))
edges[0], edges[-1] = -np.inf, np.inf
for name, data in [("2014", tr), ("2015", te)]:
    dec = pd.cut(data["score"], edges, labels=False) + 1
    g = data.groupby(dec).agg(pred=("pd", "mean"), actual=("bad", "mean"), n=("bad", "size"))
    g["year"] = name
    cal.append(g)
cal = pd.concat(cal).reset_index().rename(columns={"score": "decile"})
cal.to_csv("output/calibration.csv", index=False)
c15 = cal[cal["year"] == "2015"]
print("\n校准（按 2014 分数十等分，1 = 最低分）：预测违约率 vs 实际坏账率")
print(cal.pivot(index="decile", columns="year", values=["pred", "actual"]).round(4).to_string())
print(f"2015 整体：预测 {te['pd'].mean():.2%}，实际 {te['bad'].mean():.2%}；"
      f"10 档中有 {(c15['actual'] > c15['pred']).sum()} 档实际高于预测")
fig, ax = plt.subplots(figsize=(6, 5))
for name, mk in [("2014", "o"), ("2015", "s")]:
    g = cal[cal["year"] == name]
    ax.plot(g["pred"] * 100, g["actual"] * 100, mk + "-", label=name)
lim = max(cal["pred"].max(), cal["actual"].max()) * 100 * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Perfect calibration")
ax.set(xlabel="Predicted default rate (%)", ylabel="Actual bad rate (%)",
       title="Calibration by score decile (model fit on 2014)")
ax.legend()
plt.tight_layout()
plt.savefig("output/fig5_calibration.png", dpi=150)

# 定价诊断：按分数十等分，每档实际赚不赚钱；亏钱的档，年化利率大约要上调多少才打平
print("\n定价诊断（2015，按 2014 分数十等分）：")
dec15 = pd.cut(te["score"], edges, labels=False) + 1
pr = te.groupby(dec15).agg(n=("bad", "size"), avg_rate=("int_rate", "mean"), bad_rate=("bad", "mean"),
                           pnl=("pnl", "sum"), principal=("loan_amnt", "sum"))
pr["net_margin"] = pr["pnl"] / pr["principal"]
pr["rate_uplift_pp"] = np.maximum(0, -pr["net_margin"]) / YEARS * 100
pr.index.name = "decile"
print(pr[["n", "avg_rate", "bad_rate", "net_margin", "rate_uplift_pp"]].round(4).to_string())
under = pr[pr["net_margin"] < 0]
print(f"净收益为负的分数档：{list(under.index)}；这些档年化利率约需上调 "
      f"{under['rate_uplift_pp'].min():.1f}-{under['rate_uplift_pp'].max():.1f} 个百分点才能打平（粗略折算）")
pr.to_csv("output/pricing_by_decile.csv")

with open("output/strategy_summary.txt", "w") as fo:
    fo.write(f"selection=2014, evaluation=2015; cost K={K_BASE}, years={YEARS}\n")
    fo.write(f"single cutoff (2014-opt) {cut_opt:.1f}: 2015 approve={oos['approve']:.4f} bad={oos['bad']:.4f} profit={oos['profit']:.0f}\n")
    fo.write(f"intuitive cutoff {cut_int:.1f}: 2015 profit={intu['profit']:.0f}; approve-all profit={te['pnl'].sum():.0f}\n")
    fo.write(f"hindsight 2015 optimum {cut_hind:.1f}: profit={hind['profit']:.0f}; drift cost={hind['profit'] - oos['profit']:.0f}\n")
    fo.write(f"tier t1={t1:.1f} t2={t2:.1f}: 2015 profit={tier['profit']:.0f} lift={lift:.4f}; breakeven catch={be:.4f}\n")
    fo.write(f"tier lift range under verify assumptions: {sv['相对单阈值'].min():.4f} to {sv['相对单阈值'].max():.4f}\n")
print("\n结果表已存 output/，图已存 output/fig3、fig4、fig5")
