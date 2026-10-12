CREATE TABLE t(a, b, c); INSERT INTO t VALUES(1,'two',3.5),(NULL,'',-0.0),(1e100,'x"y',x'00ff');
.mode ascii
SELECT * FROM t;
.mode html
SELECT * FROM t;
.mode tcl
SELECT * FROM t;
.mode table
SELECT * FROM t;
.mode json
.headers on
SELECT * FROM t;
.mode csv
.nullvalue NULL
SELECT * FROM t;
.width 4 8 8
.mode column
SELECT * FROM t;
