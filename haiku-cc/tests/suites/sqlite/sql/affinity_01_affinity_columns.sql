CREATE TABLE t(i INTEGER, r REAL, t TEXT, n NUMERIC, b BLOB, x);
INSERT INTO t VALUES('12','12','12','12','12','12');
INSERT INTO t VALUES(12.0, 12, 12, 12.0, 12, 12.0);
INSERT INTO t VALUES('abc','abc','abc','abc','abc','abc');
INSERT INTO t VALUES('1e3','1e3','1e3','1e3','1e3','1e3');
INSERT INTO t VALUES(' 5','0x10','1.50','007','','');
SELECT typeof(i),typeof(r),typeof(t),typeof(n),typeof(b),typeof(x) FROM t;
SELECT * FROM t;
