SELECT sqlite_compileoption_used('ENABLE_FTS3'), sqlite_compileoption_used('ENABLE_FTS4'), sqlite_compileoption_used('ENABLE_GEOPOLY'), sqlite_compileoption_used('ENABLE_STAT4'), sqlite_compileoption_used('THREADSAFE=1');
SELECT sqlite_compileoption_get(0) IS NOT NULL, sqlite_threadsafe() IS NULL;
PRAGMA threads;
PRAGMA threads = 2;
PRAGMA threads;
CREATE TABLE t(a); WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<20000) INSERT INTO t SELECT (i*7919)%10007 FROM n;
SELECT count(*), sum(a), min(a), max(a) FROM (SELECT a FROM t ORDER BY a);
CREATE INDEX ta ON t(a); PRAGMA integrity_check;
