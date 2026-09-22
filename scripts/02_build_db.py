# Day 0: 把切片数据写进 SQLite，供 Day 1 的 SQL 风险画像使用
import sqlite3
import pandas as pd

df = pd.read_csv("data/lc_2014_2015_36m.csv.gz", low_memory=False)

con = sqlite3.connect("data/risk.db")
df.to_sql("loans", con, if_exists="replace", index=False)
con.close()

print(f"已写入 loans 表: {len(df):,} 行, {df.shape[1]} 列")
