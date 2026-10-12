CREATE TABLE t(a);
INSERT INTO t VALUES('x'),('y');
SELECT rowid, oid, _rowid_, a FROM t;
INSERT INTO t(rowid, a) VALUES(100,'z');
INSERT INTO t VALUES('w');
SELECT rowid, a FROM t;
