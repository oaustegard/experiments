SELECT 'a' = 'a  ' COLLATE RTRIM, 'a' < 'a ' COLLATE RTRIM, 'a  b' = 'a b' COLLATE RTRIM;
CREATE TABLE t(a COLLATE RTRIM); INSERT INTO t VALUES('x'),('x '),('x  '),(' x');
SELECT count(DISTINCT a) FROM t;
SELECT a, length(a) FROM t GROUP BY a ORDER BY length(a);
