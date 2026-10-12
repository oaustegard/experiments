CREATE TABLE t(a,b); INSERT INTO t VALUES(1,2),(3,4),(5,6);
SELECT * FROM t WHERE (a,b) = (3,4);
SELECT * FROM t WHERE (a,b) IN (VALUES(1,2),(5,6));
SELECT (1,2) < (1,3), (1,2) = (1,2), (2,1) > (1,9);
