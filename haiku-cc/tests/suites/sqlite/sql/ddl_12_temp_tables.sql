CREATE TEMP TABLE t(a);
INSERT INTO t VALUES(1),(2);
CREATE TABLE main.m(a);
SELECT name FROM sqlite_temp_master;
SELECT sum(a) FROM temp.t;
.tables
