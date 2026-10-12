CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,'a'),(2,'b'),(3,'c');
INSERT INTO t SELECT a+10, b||b FROM t;
SELECT * FROM t;
SELECT changes(), total_changes();
