CREATE TABLE t(v); INSERT INTO t VALUES(10),(20),(20),(30),(40),(50),(60),(70);
SELECT v, ntile(3) OVER (ORDER BY v), percent_rank() OVER (ORDER BY v), cume_dist() OVER (ORDER BY v) FROM t;
