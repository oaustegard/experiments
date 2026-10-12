CREATE TABLE t(b BLOB);
INSERT INTO t VALUES(zeroblob(100000));
INSERT INTO t VALUES(zeroblob(5));
SELECT length(b), hex(substr(b, 99999, 2)) FROM t ORDER BY length(b);
UPDATE t SET b = b || x'ff' WHERE length(b) = 5;
SELECT hex(b) FROM t WHERE length(b) < 10;
