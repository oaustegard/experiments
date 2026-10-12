CREATE TABLE t(v);
INSERT INTO t VALUES(NULL),(1),(2.5),('a'),('B'),(x'00'),(x'ff'),(-5),('10'),(1.0);
SELECT typeof(v), v FROM t ORDER BY v;
SELECT typeof(v), count(*) FROM t GROUP BY typeof(v) ORDER BY 1;
