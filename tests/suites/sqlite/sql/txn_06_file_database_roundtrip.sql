.open test_roundtrip.db
CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,'one'),(2,'two');
.open test_roundtrip.db
SELECT * FROM t;
PRAGMA integrity_check;
.open :memory:
.shell rm -f test_roundtrip.db
