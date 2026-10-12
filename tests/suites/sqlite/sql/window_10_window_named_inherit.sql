CREATE TABLE t(a,b); INSERT INTO t VALUES(1,5),(1,6),(2,7),(2,8),(2,9);
SELECT a, b, sum(b) OVER (w ORDER BY b), count(*) OVER w FROM t WINDOW w AS (PARTITION BY a) ORDER BY a, b;
