CREATE TABLE t(g,v); INSERT INTO t VALUES('a','x'),('a','y'),('b','z'),('a','w');
SELECT g, v, group_concat(v) OVER (PARTITION BY g ORDER BY rowid) FROM t ORDER BY rowid;
