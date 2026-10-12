CREATE TABLE t(a PRIMARY KEY, b); INSERT INTO t SELECT value, value*2 FROM generate_series(1, 50);
SELECT opcode FROM bytecode('SELECT a + 1 FROM t WHERE b > 3') ORDER BY addr LIMIT 6;
SELECT DISTINCT type, name FROM tables_used('SELECT * FROM t WHERE a IN (SELECT b FROM t)') ORDER BY 1, 2;
SELECT pgno, length(data) FROM sqlite_dbpage WHERE pgno <= 2 ORDER BY pgno;
SELECT sql, ncol FROM sqlite_stmt WHERE sql LIKE 'SELECT%' LIMIT 1;
