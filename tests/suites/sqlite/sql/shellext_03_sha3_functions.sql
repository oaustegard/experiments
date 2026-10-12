SELECT hex(sha3('')), hex(sha3('abc')), hex(sha3('abc', 224)), hex(sha3('abc', 384)), hex(sha3('abc', 512));
SELECT hex(sha3(x'00ff00ff')), hex(sha3(zeroblob(1000))), hex(sha3(NULL));
CREATE TABLE t(a, b); INSERT INTO t VALUES(1,'x'),(2,'y'),(3,'z');
SELECT hex(sha3_agg(a)), hex(sha3_agg(b, 256)), hex(sha3_query('SELECT * FROM t ORDER BY a', 224)) FROM t;
SELECT hex(sha3(printf('%.1000c', 'a'))), length(sha3('x', 512));
