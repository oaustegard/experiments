CREATE TABLE t(a, b);
INSERT INTO t VALUES(1, x'deadbeef'), (2, 'text'), (3, NULL);
.mode quote
SELECT * FROM t;
.output test_out.csv
.mode csv
.headers on
SELECT * FROM t;
.output stdout
.shell cat test_out.csv
CREATE TABLE u(a, b);
.import test_out.csv u --skip 1
SELECT * FROM u;
.shell rm -f test_out.csv
