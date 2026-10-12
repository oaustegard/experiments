CREATE TABLE t(id INTEGER PRIMARY KEY, b BLOB);
INSERT INTO t(b) VALUES(x'01'),(x'0100'),(x'00ff'),(zeroblob(3)),(x'');
SELECT id, hex(b), length(b) FROM t ORDER BY b, id;
SELECT max(hex(b)), min(b) IS NOT NULL FROM t;
