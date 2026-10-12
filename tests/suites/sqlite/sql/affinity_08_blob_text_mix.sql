CREATE TABLE t(a BLOB, b TEXT);
INSERT INTO t VALUES(x'48656c6c6f','Hello'),('Hello',x'48656c6c6f'),(1,1);
SELECT typeof(a), typeof(b), a, b, hex(a), hex(b), a=b, length(a), length(b) FROM t;
