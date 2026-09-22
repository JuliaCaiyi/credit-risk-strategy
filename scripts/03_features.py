# Day 1: 定义好坏标签、构造申请时点可得的特征、按时间划分训练/测试集
#
# 标签：坏 = Charged Off / Default；好 = Fully Paid；其余状态结局未定，剔除
# 表现期 = 整个 36 期贷款周期（数据截至 2018 年底，这批贷款基本都已到期）
# 时间切分：2014 年放款 -> 训练集，2015 年放款 -> 测试集（跨时间验证）
#
# 特征用白名单：只放"放款那一刻就知道"的字段，其余一律不进模型
# 被排除的典型字段及原因：
#   total_pymnt, total_rec_prncp, total_rec_int, last_pymnt_d, last_pymnt_amnt -> 贷后还款记录，放款时不存在
#   recoveries, collection_recovery_fee                                          -> 违约后才产生的催收回款
#   out_prncp, out_prncp_inv                                                     -> 剩余本金，贷后字段
#   last_fico_range_high/low, last_credit_pull_d                                 -> 贷后重新拉取的征信
#   hardship_*, settlement_*, debt_settlement_flag                               -> 出现困难后才有的记录，几乎等于答案
#   grade, sub_grade, int_rate                                                   -> LC 自己风控模型的输出，放进来等于抄答案；
#                                                                                   int_rate 保留用于 Day 3 收益计算
import numpy as np
import pandas as pd

df = pd.read_csv("data/lc_2014_2015_36m.csv.gz", low_memory=False)

bad_status = df["loan_status"].str.contains("Charged Off", na=False) | (df["loan_status"] == "Default")
good_status = df["loan_status"].str.contains("Fully Paid", na=False)
df = df[bad_status | good_status].copy()
df["bad"] = bad_status[df.index].astype(int)

df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df["issue_year"] = df["issue_dt"].dt.year
df["split"] = np.where(df["issue_year"] == 2014, "train", "test")


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


df["int_rate"] = to_num(df["int_rate"])
df["revol_util"] = to_num(df["revol_util"])

# 工作年限："< 1 year" -> 0，"10+ years" -> 10
emp = df["emp_length"].astype(str)
df["emp_length_yrs"] = emp.str.extract(r"(\d+)")[0].astype(float)
df.loc[emp.str.contains("<"), "emp_length_yrs"] = 0

# 信用账龄：放款日 - 最早信用账户开立日（月）
ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
df["credit_age_months"] = (df["issue_dt"].dt.year - ecl.dt.year) * 12 + (df["issue_dt"].dt.month - ecl.dt.month)

df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

inc = df["annual_inc"].replace(0, np.nan)
df["loan_to_income"] = df["loan_amnt"] / inc
df["installment_to_income"] = df["installment"] * 12 / inc

# 从未逾期的人这个字段是空的，空本身就是信息，转成是否有逾期史
df["ever_delinq"] = df["mths_since_last_delinq"].notna().astype(int)

NUM_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "emp_length_yrs", "fico", "credit_age_months",
    "inq_last_6mths", "delinq_2yrs", "ever_delinq", "open_acc", "total_acc", "pub_rec",
    "pub_rec_bankruptcies", "revol_bal", "revol_util", "mort_acc", "acc_open_past_24mths",
    "bc_util", "percent_bc_gt_75", "num_tl_op_past_12m", "loan_to_income", "installment_to_income",
]
CAT_FEATURES = ["home_ownership", "verification_status", "purpose"]

missing_cols = [c for c in NUM_FEATURES + CAT_FEATURES if c not in df.columns]
if missing_cols:
    raise SystemExit(f"数据里缺少这些字段: {missing_cols}")

KEEP = ["id", "issue_d", "issue_year", "split", "bad"] + NUM_FEATURES + CAT_FEATURES
df[KEEP].to_csv("data/features.csv", index=False)

# 策略层要用的金额字段单独存，不进模型
OUTCOME = ["id", "issue_year", "split", "bad", "loan_amnt", "int_rate", "installment",
           "grade", "total_pymnt", "recoveries", "collection_recovery_fee"]
df[OUTCOME].to_csv("data/loan_outcomes.csv", index=False)

print(f"有结局的贷款: {len(df):,}")
print(df.groupby("split")["bad"].agg(n="count", bad_rate="mean").round(4))
print(f"\n特征数: {len(NUM_FEATURES) + len(CAT_FEATURES)}")
print("\n特征缺失率(%):")
miss = (df[NUM_FEATURES + CAT_FEATURES].isna().mean() * 100).round(2)
print(miss[miss > 0].sort_values(ascending=False).to_string())
print("\n数值特征概览:")
print(df[NUM_FEATURES].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())
