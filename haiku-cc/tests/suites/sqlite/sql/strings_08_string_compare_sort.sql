CREATE TABLE t(s); INSERT INTO t VALUES('b'),('B'),('a'),('A'),('10'),('9'),('é'),('_'),(''),(NULL);
SELECT s FROM t ORDER BY s;
SELECT s FROM t ORDER BY s COLLATE NOCASE, s;
SELECT s FROM t ORDER BY s DESC;
SELECT min(s), max(s) FROM t;
