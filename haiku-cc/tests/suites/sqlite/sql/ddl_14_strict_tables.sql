CREATE TABLE t(a INTEGER, b TEXT, c REAL, d ANY) STRICT;
INSERT INTO t VALUES(1,'x',1.5,'any');
INSERT INTO t VALUES('abc','x',1.5,2);
INSERT INTO t VALUES(1,2,'3.5',x'00');
SELECT *, typeof(a), typeof(b), typeof(c) FROM t;
