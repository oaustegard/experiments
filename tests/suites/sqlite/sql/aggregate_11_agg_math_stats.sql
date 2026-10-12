CREATE TABLE t(x);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<100) INSERT INTO t SELECT i*1.5 FROM n;
SELECT count(*), sum(x), avg(x), min(x), max(x), sum(x*x), round(avg(x*x)-avg(x)*avg(x),4) FROM t;
