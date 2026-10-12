CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,1),(1,2),(2,2),(2,2),(3,NULL);
SELECT count(DISTINCT a), count(DISTINCT b), sum(DISTINCT a), sum(DISTINCT b), group_concat(DISTINCT a) FROM t;
