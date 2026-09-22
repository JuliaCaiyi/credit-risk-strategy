# Day 0: 从全量 Lending Club 数据中切出 2014-2015 年放款、36 期的贷款
# 选这段的原因：数据截止 2018 年底，这批 36 期贷款基本都已到期，每笔都能看到最终结局
import pandas as pd

SRC = "data/archive.zip"   # Kaggle 下载的压缩包，不用解压，pandas 能直接读
OUT = "data/lc_2014_2015_36m.csv.gz"

# 原表 150 多列，只留后面用得到的，省内存也省时间
COLS = [
    "id", "issue_d", "term", "loan_status",
    # 申请时点可得：用于建模
    "loan_amnt", "installment", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "acc_open_past_24mths",
    "bc_util", "percent_bc_gt_75", "num_tl_op_past_12m",
    # LC 自己的定价结果：不进模型，策略层算收益用
    "int_rate", "grade", "sub_grade",
    # 贷后结果：不进模型，策略层算真实损失用
    "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee",
]

print("开始读取，全量约 226 万行，分块处理，每块 20 万行……")
pieces = []
n_read = 0
for chunk in pd.read_csv(SRC, usecols=COLS, chunksize=200_000, low_memory=False):
    n_read += len(chunk)
    year = chunk["issue_d"].astype(str).str[-4:]
    term = chunk["term"].astype(str).str.strip()
    keep = chunk[year.isin(["2014", "2015"]) & (term == "36 months")]
    pieces.append(keep)
    print(f"已读 {n_read:,} 行，保留 {sum(len(p) for p in pieces):,} 行")

df = pd.concat(pieces, ignore_index=True)
df.to_csv(OUT, index=False)

print("\n切片完成:", df.shape)
print("\n按年份:")
print(df["issue_d"].str[-4:].value_counts())
print("\n贷款状态分布:")
print(df["loan_status"].value_counts())
