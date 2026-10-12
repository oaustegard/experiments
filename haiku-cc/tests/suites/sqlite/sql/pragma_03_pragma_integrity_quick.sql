CREATE TABLE t(a PRIMARY KEY, b); CREATE INDEX ib ON t(b);
INSERT INTO t SELECT value, value*2 FROM json_each('[1,2,3,4,5,6,7,8]');
PRAGMA integrity_check;
PRAGMA quick_check;
SELECT page_count > 0 FROM pragma_page_count;
PRAGMA freelist_count;
PRAGMA optimize;
ANALYZE;
SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY 1,2;
