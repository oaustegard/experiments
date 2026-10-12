CREATE TABLE t(a);
SELECT count(*), sum(a), total(a), avg(a), min(a), max(a), group_concat(a) FROM t;
SELECT a, count(*) FROM t GROUP BY a;
