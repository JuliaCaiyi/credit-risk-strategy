# Day 3: 策略层 —— 阈值优化、三层分层策略、策略评估体系
#
# 收益口径（每笔贷款，用真实还款现金流回测）：
#   净收益 = 实际收回总额 total_pymnt - 放款本金 - 资金与运营成本
#   资金与运营成本 = 本金 × 年化成本率 K × 平均占用年限 1.5（36 期等额本息，平均未还本金约为一半，占用约 1.5 年）
#   好客户：净收益 = 实收利息 - 成本；坏客户：净收益 = 违约前已收回的钱 - 本金 - 成本（通常为负）
#   被拒绝：收益为 0
# 策略回测用 2015 年贷款（评分卡训练时没见过的数据）
# 注意：这里的"通过率"是在 LC 已放款客群中的通过率，LC 拒掉的申请人不在数据里
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_BASE = 0.03
YEARS = 1.5
# 中间层"加验"（人脸 / 补充收入证明等）的假设参数
CATCH_BAD = 0.40    # 加验能拦下的坏客户比例
DROP_GOOD = 0.10    # 加验导致好客户流失的比例（体验损伤）
VERIFY_COST = 30    # 每笔加验成本（美元）
V_MAX = 0.20        # 加验比例上限（体验约束）

s = pd.read_csv("data/scored.csv")
o = pd.read_csv("data/loan_outcomes.csv")[["id", "loan_amnt", "total_pymnt"]]
f = pd.read_csv("data/features.csv")[["id", "issue_d", "fico", "dti", "inq_last_6mths"]]
d = s.merge(o, on="id").merge(f, on="id")


def add_pnl(data, k):
    return data["total_pymnt"] - data["loan_amnt"] * (1 + k * YEARS)


d["pnl"] = add_pnl(d, K_BASE)
te = d[d["split"] == "test"].sort_values("score", ascending=False).reset_index(drop=True)
tr = d[d["split"] == "train"].sort_values("score", ascending=False).reset_index(drop=True)
N = len(te)

# ---------- 3.1 单阈值：从高分往低分依次放行，逐一计算 ----------
cum_pnl = te["pnl"].cumsum().values
cum_bad = te["bad"].cumsum().values
approve_rate = np.arange(1, N + 1) / N
bad_rate = cum_bad / np.arange(1, N + 1)
j_opt = cum_pnl.argmax()
opt = {"approve": approve_rate[j_opt], "bad": bad_rate[j_opt],
       "profit": cum_pnl[j_opt], "score": te["score"][j_opt]}
print("== 3.1 单阈值优化（2015 回测，年化成本 3%）==")
print(f"收益最大阈值: 分数 >= {opt['score']:.1f}  通过率 {opt['approve']:.1%}  "
      f"通过客群坏账率 {opt['bad']:.2%}  净收益 ${opt['profit'] / 1e6:.1f}M")
print(f"全部通过: 坏账率 {bad_rate[-1]:.2%}  净收益 ${cum_pnl[-1] / 1e6:.1f}M")

TARGET_BAD = 0.10
j_int = np.where(bad_rate <= TARGET_BAD)[0].max()
print(f"风控直觉阈值（通过客群坏账率 <= 10%）: 通过率 {approve_rate[j_int]:.1%}  "
      f"净收益 ${cum_pnl[j_int] / 1e6:.1f}M，比最优少 ${(cum_pnl[j_opt] - cum_pnl[j_int]) / 1e6:.1f}M")

print("\n通过率每提高 1 个百分点（放行下一批 1% 的客户）:")
marg = []
for a in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
    i0, i1 = int(N * a) - 1, int(N * (a + 0.01)) - 1
    marg.append({"approve_from": f"{a:.0%}", "slice_bad_rate": te["bad"][i0 + 1:i1 + 1].mean(),
                 "overall_bad_rate_change_pp": (bad_rate[i1] - bad_rate[i0]) * 100,
                 "profit_change_$K": (cum_pnl[i1] - cum_pnl[i0]) / 1e3})
marg = pd.DataFrame(marg)
print(marg.round(3).to_string(index=False))
marg.to_csv("output/marginal_approval.csv", index=False)

print("\n成本敏感性（年化资金与运营成本率 K 不同时的最优通过率）:")
sens = []
for k in [0.0, 0.02, 0.03, 0.04, 0.05]:
    c = add_pnl(te, k).cumsum().values
    j = c.argmax()
    sens.append({"K": k, "opt_approve_rate": approve_rate[j], "bad_rate": bad_rate[j],
                 "profit_$M": c[j] / 1e6, "approve_all_profit_$M": c[-1] / 1e6})
sens = pd.DataFrame(sens)
print(sens.round(4).to_string(index=False))
sens.to_csv("output/cost_sensitivity.csv", index=False)

# 图 3：阈值 - 通过率 - 坏账率 - 净收益
fig, ax1 = plt.subplots(figsize=(9, 5))
step = slice(None, None, 200)
ax1.plot(te["score"][step], approve_rate[step] * 100, label="Approval rate (%)")
ax1.plot(te["score"][step], bad_rate[step] * 100, label="Bad rate of approved (%)")
ax1.set(xlabel="Score cutoff (approve if score >= cutoff)", ylabel="%")
ax2 = ax1.twinx()
ax2.plot(te["score"][step], cum_pnl[step] / 1e6, color="green", label="Net profit ($M)")
ax2.axvline(opt["score"], color="red", ls="--", lw=1)
ax2.set_ylabel("Net profit ($M)")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center left")
ax1.set_title(f"Cutoff trade-off (2015, cost {K_BASE:.0%}/yr): optimum at score {opt['score']:.0f}, "
              f"approval {opt['approve']:.0%}")
plt.tight_layout()
plt.savefig("output/fig3_cutoff_tradeoff.png", dpi=150)

# ---------- 3.2 三层分层策略：拒绝 / 加验 / 直接通过，两个阈值网格搜索 ----------
asc = te.sort_values("score").reset_index(drop=True)
grid = np.unique(np.r_[0, (np.linspace(0, 1, 201) * N).astype(int), N])
is_bad = asc["bad"].values
csum = lambda v: np.r_[0, np.cumsum(v)][grid]
C_n = grid.astype(float)
C_good_pnl = csum(np.where(is_bad == 0, asc["pnl"], 0))
C_bad_pnl = csum(np.where(is_bad == 1, asc["pnl"], 0))
C_good = csum(1 - is_bad)
C_bad = csum(is_bad)
TOT = lambda c: c[-1]

# 下标 i <= j：分数最低的 i 段拒绝，[i, j) 段加验，j 以上直接通过
I, J = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
valid = I <= J
mid_n = C_n[J] - C_n[I]
mid_good, mid_bad = C_good[J] - C_good[I], C_bad[J] - C_bad[I]
mid_profit = ((C_good_pnl[J] - C_good_pnl[I]) * (1 - DROP_GOOD)
              + (C_bad_pnl[J] - C_bad_pnl[I]) * (1 - CATCH_BAD) - VERIFY_COST * mid_n)
top_profit = TOT(C_good_pnl) + TOT(C_bad_pnl) - C_good_pnl[J] - C_bad_pnl[J]
profit = np.where(valid, mid_profit + top_profit, -np.inf)
approved = (N - C_n[J]) + mid_good * (1 - DROP_GOOD) + mid_bad * (1 - CATCH_BAD)
approved_bad = (TOT(C_bad) - C_bad[J]) + mid_bad * (1 - CATCH_BAD)
appr_rate = approved / N
verify_rate = mid_n / N


def solve(p_min, tiered=True, exact=False):
    ok = valid & (appr_rate >= p_min) & (verify_rate <= (V_MAX if tiered else 0))
    if exact:  # 前沿曲线：把通过率固定在 p 附近（±1.25 个百分点），看每个通过率水平下最多能赚多少
        ok &= appr_rate < p_min + 0.025
    if not ok.any():
        return None
    k = np.where(ok, profit, -np.inf).argmax()
    i, j = np.unravel_index(k, profit.shape)
    return i, j


def describe(i, j):
    score_at = lambda g: asc["score"].iloc[min(grid[g], N - 1)]
    return {"t1": score_at(i) if grid[i] > 0 else -np.inf, "t2": score_at(j) if grid[j] < N else np.inf,
            "reject": grid[i] / N, "verify": verify_rate[i, j], "approve": appr_rate[i, j],
            "bad_rate": approved_bad[i, j] / approved[i, j], "profit": profit[i, j]}


P_BASE = 0.85
i3, j3 = solve(P_BASE, tiered=True)
i1, j1 = solve(P_BASE, tiered=False)
tier, single = describe(i3, j3), describe(i1, j1)
lift = tier["profit"] / single["profit"] - 1
print(f"\n== 3.2 约束下的三层策略（通过率 >= {P_BASE:.0%}，加验比例 <= {V_MAX:.0%}）==")
print(f"单阈值: 通过率 {single['approve']:.1%}  坏账率 {single['bad_rate']:.2%}  净收益 ${single['profit'] / 1e6:.1f}M")
print(f"三层  : 通过率 {tier['approve']:.1%}  坏账率 {tier['bad_rate']:.2%}  净收益 ${tier['profit'] / 1e6:.1f}M"
      f"  （相对单阈值 +{lift:.1%}）")

# 加验参数是假设值，看三层策略的提升对它们有多敏感
sens_v = []
for cb in [0.2, 0.3, 0.4]:
    for dg in [0.05, 0.10, 0.20]:
        mp = ((C_good_pnl[J] - C_good_pnl[I]) * (1 - dg) + (C_bad_pnl[J] - C_bad_pnl[I]) * (1 - cb)
              - VERIFY_COST * mid_n)
        pr = np.where(valid, mp + top_profit, -np.inf)
        ap = ((N - C_n[J]) + mid_good * (1 - dg) + mid_bad * (1 - cb)) / N
        ok = valid & (ap >= P_BASE) & (verify_rate <= V_MAX)
        best = np.where(ok, pr, -np.inf).max()
        sens_v.append({"catch_bad": cb, "drop_good": dg, "tier_profit_$M": best / 1e6,
                       "lift_vs_single": best / single["profit"] - 1})
sens_v = pd.DataFrame(sens_v)
print("\n加验参数敏感性（三层策略相对单阈值的提升）:")
print(sens_v.round(3).to_string(index=False))
sens_v.to_csv("output/verify_sensitivity.csv", index=False)

layers = []
for name, lo, hi, action in [("L3 Reject", -np.inf, tier["t1"], "拒绝"),
                             ("L2 Verify", tier["t1"], tier["t2"], "加验后放款"),
                             ("L1 Approve", tier["t2"], np.inf, "直接通过")]:
    seg = te[(te["score"] >= lo) & (te["score"] < hi)]
    layers.append({"layer": name, "score_range": f"[{lo:.0f}, {hi:.0f})", "share": len(seg) / N,
                   "bad_rate_before_action": seg["bad"].mean(), "action": action})
layers = pd.DataFrame(layers)
print("\n三层分层策略表:")
print(layers.round(4).to_string(index=False))
layers.to_csv("output/tier_strategy.csv", index=False)

front = []
for p in np.arange(0.60, 0.951, 0.025):
    r3, r1 = solve(p, True, exact=True), solve(p, False, exact=True)
    front.append({"approve_min": round(p, 3),
                  "single_profit_$M": describe(*r1)["profit"] / 1e6 if r1 else np.nan,
                  "single_bad_rate": describe(*r1)["bad_rate"] if r1 else np.nan,
                  "tier_profit_$M": describe(*r3)["profit"] / 1e6 if r3 else np.nan,
                  "tier_bad_rate": describe(*r3)["bad_rate"] if r3 else np.nan})
front = pd.DataFrame(front)
print("\n风险-收益前沿（通过率从 60% 逐步放宽到 95%）:")
print(front.round(4).to_string(index=False))
front.to_csv("output/risk_return_frontier.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(front["single_bad_rate"] * 100, front["single_profit_$M"], "o-", label="Single cutoff")
ax.plot(front["tier_bad_rate"] * 100, front["tier_profit_$M"], "s-", label="3-tier (with verification)")
for _, r in front.iterrows():
    ax.annotate(f"{r['approve_min']:.0%}", (r["tier_bad_rate"] * 100, r["tier_profit_$M"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set(xlabel="Bad rate of approved (%)", ylabel="Net profit ($M)",
       title="Risk-return frontier (labels = approval rate)")
ax.legend()
plt.tight_layout()
plt.savefig("output/fig4_risk_return_frontier.png", dpi=150)

# ---------- 3.3 策略评估体系 ----------
print("\n== 3.3 策略评估 ==")
# Swap Set：旧策略 = 只看 FICO 的规则；新策略 = 评分卡；两者通过率相同
rate = opt["approve"]
new_ok = te["score"] >= te["score"].quantile(1 - rate)
old_ok = te["fico"] >= te["fico"].quantile(1 - rate)
groups = {"都通过": new_ok & old_ok, "Swap in（旧拒新过）": new_ok & ~old_ok,
          "Swap out（旧过新拒）": ~new_ok & old_ok, "都拒绝": ~new_ok & ~old_ok}
swap = pd.DataFrame([{"group": g, "n": m.sum(), "share": m.mean(), "bad_rate": te.loc[m, "bad"].mean()}
                     for g, m in groups.items()])
print(f"Swap Set（旧：只看 FICO；新：评分卡；通过率都设为 {rate:.1%}）:")
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
    ok = g["score"] >= opt["score"]
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

# 用 2014 数据定的阈值直接用在 2015，会差多少
c_tr = tr["pnl"].cumsum().values
cut14 = tr["score"][c_tr.argmax()]
p_14 = te.loc[te["score"] >= cut14, "pnl"].sum()
print(f"\n阈值漂移检查: 2014 年最优阈值 {cut14:.1f} 用在 2015 -> 通过率 {(te['score'] >= cut14).mean():.1%}，"
      f"净收益 ${p_14 / 1e6:.1f}M；2015 自身最优 ${opt['profit'] / 1e6:.1f}M（差 ${(opt['profit'] - p_14) / 1e6:.1f}M）")

with open("output/strategy_summary.txt", "w") as fo:
    fo.write(f"cost K={K_BASE}, years={YEARS}; verify catch={CATCH_BAD}, drop={DROP_GOOD}, cost=${VERIFY_COST}\n")
    fo.write(f"single optimum: score>={opt['score']:.1f} approve={opt['approve']:.4f} bad={opt['bad']:.4f} profit={opt['profit']:.0f}\n")
    fo.write(f"intuitive (bad<=10%): approve={approve_rate[j_int]:.4f} profit={cum_pnl[j_int]:.0f}\n")
    fo.write(f"constrained approve>={P_BASE}: single={single['profit']:.0f} tier={tier['profit']:.0f} lift={lift:.4f}\n")
    fo.write(f"tier thresholds: t1={tier['t1']:.1f} t2={tier['t2']:.1f} verify={tier['verify']:.4f}\n")
    fo.write(f"2014 cutoff on 2015: profit={p_14:.0f} vs optimum {opt['profit']:.0f}\n")
print("\n结果表已存 output/，图已存 output/fig3_cutoff_tradeoff.png, output/fig4_risk_return_frontier.png")
