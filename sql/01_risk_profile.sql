-- 业务问题：这批贷款整体坏账率多少？哪些维度能把好客户和坏客户区分开？
-- 运行：sqlite3 data/risk.db < sql/01_risk_profile.sql
.headers on
.mode column

-- 标签口径：坏 = 核销 / 违约；好 = 已还清；其余（仍在还款、逾期中）结局未定，剔除
DROP VIEW IF EXISTS labeled;
CREATE VIEW labeled AS
SELECT *,
       CASE
         WHEN loan_status LIKE '%Charged Off%' OR loan_status = 'Default' THEN 1
         WHEN loan_status LIKE '%Fully Paid%' THEN 0
       END AS bad,
       CAST(REPLACE(int_rate, '%', '') AS REAL)   AS rate,
       CAST(REPLACE(revol_util, '%', '') AS REAL) AS rutil
FROM loans;

-- 1. 状态分布与标签映射
SELECT loan_status, bad, COUNT(*) AS n
FROM labeled GROUP BY loan_status, bad ORDER BY n DESC;

-- 2. 整体坏账率（只算有结局的）
SELECT COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL;

-- 3. 按放款年份（vintage）
SELECT substr(issue_d, -4) AS issue_year, COUNT(*) AS n,
       ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL GROUP BY issue_year;

-- 4. 按 LC 评级：检验坏账率是否随评级单调上升
SELECT grade, COUNT(*) AS n, ROUND(AVG(rate), 2) AS avg_rate,
       ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL GROUP BY grade ORDER BY grade;

-- 5. 按 FICO 分段
SELECT CASE
         WHEN fico_range_low < 680 THEN '1: <680'
         WHEN fico_range_low < 700 THEN '2: 680-699'
         WHEN fico_range_low < 720 THEN '3: 700-719'
         WHEN fico_range_low < 750 THEN '4: 720-749'
         ELSE '5: 750+'
       END AS fico_band,
       COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL GROUP BY fico_band ORDER BY fico_band;

-- 6. 按负债收入比 DTI 十等分（窗口函数 NTILE）
WITH d AS (
  SELECT bad, dti, NTILE(10) OVER (ORDER BY dti) AS dti_decile
  FROM labeled WHERE bad IS NOT NULL AND dti IS NOT NULL
)
SELECT dti_decile, ROUND(MIN(dti), 1) AS dti_min, ROUND(MAX(dti), 1) AS dti_max,
       COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM d GROUP BY dti_decile ORDER BY dti_decile;

-- 7. 按借款用途（样本 >= 1000）
SELECT purpose, COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL
GROUP BY purpose HAVING n >= 1000 ORDER BY bad_rate_pct DESC;

-- 8. 按住房情况
SELECT home_ownership, COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL
GROUP BY home_ownership HAVING n >= 100 ORDER BY bad_rate_pct DESC;

-- 9. 按近 6 个月征信查询次数（查询越多通常越缺钱）
SELECT CASE WHEN inq_last_6mths >= 3 THEN '3+' ELSE CAST(CAST(inq_last_6mths AS INT) AS TEXT) END AS inq_6m,
       COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL AND inq_last_6mths IS NOT NULL
GROUP BY inq_6m ORDER BY inq_6m;

-- 10. 按贷款金额分段
SELECT CASE
         WHEN loan_amnt < 5000  THEN '1: <5k'
         WHEN loan_amnt < 10000 THEN '2: 5k-10k'
         WHEN loan_amnt < 20000 THEN '3: 10k-20k'
         ELSE '4: 20k+'
       END AS amt_band,
       COUNT(*) AS n, ROUND(AVG(bad) * 100, 2) AS bad_rate_pct
FROM labeled WHERE bad IS NOT NULL GROUP BY amt_band ORDER BY amt_band;

-- 11. 关键字段缺失率
SELECT COUNT(*) AS n,
       ROUND(100.0 * SUM(emp_length IS NULL) / COUNT(*), 2)             AS emp_length_miss,
       ROUND(100.0 * SUM(dti IS NULL) / COUNT(*), 2)                    AS dti_miss,
       ROUND(100.0 * SUM(revol_util IS NULL) / COUNT(*), 2)             AS revol_util_miss,
       ROUND(100.0 * SUM(mths_since_last_delinq IS NULL) / COUNT(*), 2) AS last_delinq_miss,
       ROUND(100.0 * SUM(mort_acc IS NULL) / COUNT(*), 2)               AS mort_acc_miss,
       ROUND(100.0 * SUM(bc_util IS NULL) / COUNT(*), 2)                AS bc_util_miss
FROM labeled WHERE bad IS NOT NULL;
