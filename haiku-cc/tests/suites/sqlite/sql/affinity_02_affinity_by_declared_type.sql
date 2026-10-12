CREATE TABLE t(a INT, b VARCHAR(10), c DOUBLE, d DECIMAL(5,2), e BOOLEAN, f DATETIME, g CLOB, h FLOATING POINT, i UNSIGNED BIG INT, j NCHAR(5));
INSERT INTO t VALUES('1','1',1,'1.5',1,'2024','x','2.5','7',9);
SELECT typeof(a),typeof(b),typeof(c),typeof(d),typeof(e),typeof(f),typeof(g),typeof(h),typeof(i),typeof(j) FROM t;
