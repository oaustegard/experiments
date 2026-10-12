CREATE TABLE t(s); INSERT INTO t VALUES('a10'),('a9'),('a100'),('A2'),('a1'),('b'),('a02');
SELECT s FROM t ORDER BY s COLLATE uint;
SELECT s FROM t ORDER BY s;
SELECT 'x10' < 'x9' COLLATE uint, 'x10' < 'x9', 'a1b' = 'a01b' COLLATE uint;
