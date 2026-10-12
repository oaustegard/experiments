# SQL test scripts, part C: query planner output (EXPLAIN / EXPLAIN QUERY PLAN), the
# extensions compiled into shell.c (series, regexp, sha3, decimal, ieee754, base64, uint,
# zipfile, fileio, dbdata, completion, ...) and virtual tables built into the amalgamation.
# SCRIPTS_EXT holds scripts that need the extended build config (fts3/fts4, geopoly, stat4,
# threads, ...): written to sql-ext/ and run only in the `ext` configuration.

SCHEMA = """
CREATE TABLE t1(a INTEGER PRIMARY KEY, b TEXT, c REAL, d);
CREATE TABLE t2(x INTEGER, y TEXT, z);
CREATE TABLE t3(k TEXT PRIMARY KEY, v) WITHOUT ROWID;
CREATE INDEX t1b ON t1(b);
CREATE INDEX t1cd ON t1(c, d);
CREATE UNIQUE INDEX t2x ON t2(x);
CREATE INDEX t2yz ON t2(y, z) WHERE z > 0;
"""

SCRIPTS = {}

SCRIPTS["explain"] = [
("eqp_simple_scan", SCHEMA + """
EXPLAIN QUERY PLAN SELECT * FROM t1;
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE a = 5;
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE a > 5 AND a < 10;
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE b = 'x';
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE b LIKE 'abc%';
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE c = 1.5 AND d = 2;
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE d = 2;
"""),
("eqp_joins", SCHEMA + """
EXPLAIN QUERY PLAN SELECT * FROM t1 JOIN t2 ON t1.a = t2.x;
EXPLAIN QUERY PLAN SELECT * FROM t1, t2 WHERE t1.b = t2.y AND t2.z > 0;
EXPLAIN QUERY PLAN SELECT * FROM t1 LEFT JOIN t2 ON t1.a = t2.x LEFT JOIN t3 ON t3.k = t2.y;
EXPLAIN QUERY PLAN SELECT * FROM t1 CROSS JOIN t2 CROSS JOIN t3;
EXPLAIN QUERY PLAN SELECT * FROM t3 WHERE k = 'a' OR k = 'b';
EXPLAIN QUERY PLAN SELECT count(*) FROM t1, t2, t3 WHERE t1.a = t2.x AND t2.y = t3.k;
"""),
("eqp_order_group", SCHEMA + """
EXPLAIN QUERY PLAN SELECT * FROM t1 ORDER BY a;
EXPLAIN QUERY PLAN SELECT * FROM t1 ORDER BY b;
EXPLAIN QUERY PLAN SELECT * FROM t1 ORDER BY c, d;
EXPLAIN QUERY PLAN SELECT * FROM t1 ORDER BY d;
EXPLAIN QUERY PLAN SELECT b, count(*) FROM t1 GROUP BY b;
EXPLAIN QUERY PLAN SELECT d, count(*) FROM t1 GROUP BY d ORDER BY 2;
EXPLAIN QUERY PLAN SELECT DISTINCT b FROM t1;
EXPLAIN QUERY PLAN SELECT max(a), min(b) FROM t1;
"""),
("eqp_subqueries", SCHEMA + """
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE a IN (SELECT x FROM t2);
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.x = t1.a);
EXPLAIN QUERY PLAN SELECT (SELECT max(x) FROM t2) + a FROM t1;
EXPLAIN QUERY PLAN SELECT * FROM (SELECT * FROM t1 WHERE a > 3) WHERE b = 'x';
EXPLAIN QUERY PLAN WITH c AS MATERIALIZED (SELECT * FROM t1) SELECT * FROM c, c AS d WHERE c.a = d.a;
EXPLAIN QUERY PLAN SELECT a FROM t1 UNION SELECT x FROM t2 ORDER BY 1;
EXPLAIN QUERY PLAN SELECT a FROM t1 INTERSECT SELECT x FROM t2;
"""),
("eqp_covering_and_partial", SCHEMA + """
EXPLAIN QUERY PLAN SELECT b FROM t1 WHERE b > 'm';
EXPLAIN QUERY PLAN SELECT c, d FROM t1 WHERE c < 2.0;
EXPLAIN QUERY PLAN SELECT y FROM t2 WHERE y = 'q' AND z > 5;
EXPLAIN QUERY PLAN SELECT y FROM t2 WHERE y = 'q' AND z < 0;
EXPLAIN QUERY PLAN SELECT * FROM t2 WHERE x BETWEEN 3 AND 9;
EXPLAIN QUERY PLAN SELECT * FROM t2 WHERE x IN (1,2,3);
EXPLAIN QUERY PLAN SELECT * FROM t2 INDEXED BY t2x WHERE x = 1;
EXPLAIN QUERY PLAN SELECT * FROM t2 NOT INDEXED WHERE x = 1;
"""),
("eqp_after_analyze", SCHEMA + """
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<500)
INSERT INTO t1 SELECT i, 'b'||(i%10), i*0.5, i%7 FROM n;
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<300)
INSERT INTO t2 SELECT i, 'y'||(i%5), i-100 FROM n;
ANALYZE;
SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY 1, 2;
EXPLAIN QUERY PLAN SELECT * FROM t1 WHERE b = 'b3' AND d = 2;
EXPLAIN QUERY PLAN SELECT * FROM t1 JOIN t2 ON t1.d = t2.x WHERE t1.b = 'b1';
EXPLAIN QUERY PLAN SELECT t2.y, count(*) FROM t2 JOIN t1 ON t1.a = t2.x GROUP BY t2.y;
"""),
("explain_bytecode_simple", """
CREATE TABLE t(a, b);
EXPLAIN SELECT a + b FROM t WHERE a > 1;
EXPLAIN INSERT INTO t VALUES(1, 'x');
EXPLAIN UPDATE t SET b = 'y' WHERE a = 1;
EXPLAIN DELETE FROM t WHERE a = 1;
"""),
("explain_bytecode_agg_sort", """
CREATE TABLE t(a, b, c);
CREATE INDEX tb ON t(b);
EXPLAIN SELECT b, count(*), sum(c) FROM t GROUP BY b ORDER BY 2 DESC LIMIT 3;
EXPLAIN SELECT DISTINCT a FROM t WHERE b IN (1,2,3);
EXPLAIN SELECT a FROM t UNION ALL SELECT b FROM t ORDER BY 1;
"""),
("explain_bytecode_window_cte", """
CREATE TABLE t(g, v);
EXPLAIN SELECT g, v, sum(v) OVER (PARTITION BY g ORDER BY v ROWS 2 PRECEDING), rank() OVER (ORDER BY v) FROM t;
EXPLAIN WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<5) SELECT n FROM c;
EXPLAIN SELECT json_extract('{"a":1}', '$.a'), printf('%d', 5), strftime('%Y', '2024-01-01');
"""),
("explain_bytecode_triggers_fk", """
PRAGMA foreign_keys = ON;
CREATE TABLE p(id INTEGER PRIMARY KEY, n);
CREATE TABLE c(id INTEGER PRIMARY KEY, p REFERENCES p(id) ON DELETE CASCADE ON UPDATE SET NULL);
CREATE TABLE log(m);
CREATE TRIGGER tr AFTER INSERT ON c BEGIN INSERT INTO log VALUES('new ' || new.id); END;
EXPLAIN INSERT INTO c VALUES(1, 1);
EXPLAIN DELETE FROM p WHERE id = 1;
EXPLAIN UPDATE p SET id = id + 1;
"""),
("planner_cost_choices", """
CREATE TABLE big(a, b, c);
CREATE TABLE small(a, b);
CREATE INDEX bia ON big(a);
CREATE INDEX bib ON big(b);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<2000) INSERT INTO big SELECT i, i%50, i%3 FROM n;
INSERT INTO small VALUES(1,1),(2,2),(3,3);
ANALYZE;
EXPLAIN QUERY PLAN SELECT * FROM big, small WHERE big.a = small.a;
EXPLAIN QUERY PLAN SELECT * FROM big WHERE a > 1990 AND b = 4;
EXPLAIN QUERY PLAN SELECT * FROM big WHERE b = 4 AND c = 1;
SELECT count(*) FROM big, small WHERE big.b = small.b;
SELECT b, count(*) FROM big WHERE a BETWEEN 100 AND 400 GROUP BY b ORDER BY 2 DESC, 1 LIMIT 3;
SELECT stat FROM sqlite_stat1 WHERE idx = 'bib';
"""),
]

SCRIPTS["shellext"] = [
("generate_series", """
SELECT value FROM generate_series(1, 5);
SELECT value FROM generate_series(10, 0, -3);
SELECT sum(value), count(*) FROM generate_series(1, 1000);
SELECT value FROM generate_series(0, 1, 0.25);
SELECT a.value * b.value FROM generate_series(1,3) a, generate_series(1,3) b WHERE a.value <= b.value;
SELECT value FROM generate_series(5, 1);
SELECT group_concat(value) FROM generate_series(1, 20, 7);
"""),
("regexp_function", """
SELECT 'hello' REGEXP 'h.*o', 'hello' REGEXP '^e', 'a1b2' REGEXP '[0-9]+', 'ABC' REGEXP '[a-c]+', 'foo.bar' REGEXP 'o\\.b';
CREATE TABLE t(s); INSERT INTO t VALUES('apple pie'),('banana'),('cherry tart'),('date');
SELECT s FROM t WHERE s REGEXP '(an|er)+' ORDER BY s;
SELECT s FROM t WHERE s REGEXP '^[a-c].* [pt]' ORDER BY s;
SELECT regexp('a+b', 'caaab'), regexp('^$', ''), regexp('\\d{3}', 'ab123');
"""),
("sha3_functions", """
SELECT hex(sha3('')), hex(sha3('abc')), hex(sha3('abc', 224)), hex(sha3('abc', 384)), hex(sha3('abc', 512));
SELECT hex(sha3(x'00ff00ff')), hex(sha3(zeroblob(1000))), hex(sha3(NULL));
CREATE TABLE t(a, b); INSERT INTO t VALUES(1,'x'),(2,'y'),(3,'z');
SELECT hex(sha3_agg(a)), hex(sha3_agg(b, 256)), hex(sha3_query('SELECT * FROM t ORDER BY a', 224)) FROM t;
SELECT hex(sha3(printf('%.1000c', 'a'))), length(sha3('x', 512));
"""),
("decimal_extension", """
SELECT decimal_add('1.1', '2.2'), decimal_sub('10', '0.001'), decimal_mul('123456789.123456789', '987654321.987654321'), decimal('0.1e3'), decimal('-0.0050');
SELECT decimal_cmp('1.0', '1'), decimal_cmp('2', '10'), decimal_cmp('-1', '-2'), decimal_pow2(10), decimal_exp(decimal('1e2')) IS NOT NULL;
SELECT decimal_sum(value) FROM (SELECT '0.1' AS value UNION ALL SELECT '0.2' UNION ALL SELECT '0.3');
SELECT decimal_mul(decimal_add('1e50', '1'), '3'), decimal_sub('1', '2e-30'), decimal_add('', '1'), decimal_add('abc', '1');
SELECT 'a' < 'b' COLLATE decimal, '10' < '9' COLLATE decimal, '1.50' = '1.5' COLLATE decimal;
"""),
("ieee754_functions", """
SELECT ieee754(0.5), ieee754(-1.25), ieee754(1e100), ieee754(0.1), ieee754_mantissa(0.5), ieee754_exponent(8.0);
SELECT ieee754_to_blob(1.0) IS NOT NULL, hex(ieee754_to_blob(-2.5)), ieee754_from_blob(ieee754_to_blob(3.14159)), ieee754(5, 10), ieee754(1, -1074);
SELECT ieee754_from_blob(x'3ff0000000000000'), ieee754_from_blob(x'7ff0000000000000'), ieee754(3), ieee754(NULL);
"""),
("uint_collation", """
CREATE TABLE t(s); INSERT INTO t VALUES('a10'),('a9'),('a100'),('A2'),('a1'),('b'),('a02');
SELECT s FROM t ORDER BY s COLLATE uint;
SELECT s FROM t ORDER BY s;
SELECT 'x10' < 'x9' COLLATE uint, 'x10' < 'x9', 'a1b' = 'a01b' COLLATE uint;
"""),
("fileio_functions", """
SELECT writefile('qjs_fileio.txt', 'line1' || char(10) || 'line2');
SELECT readfile('qjs_fileio.txt'), length(readfile('qjs_fileio.txt'));
SELECT writefile('qjs_fileio.bin', x'00010203');
SELECT hex(readfile('qjs_fileio.bin')), readfile('nonexistent.file') IS NULL;
SELECT count(*) FROM fsdir('.') WHERE name LIKE 'qjs_fileio%';
SELECT lsmode(33188), lsmode(16877);
SELECT name, mode FROM fsdir('.') WHERE name = 'qjs_fileio.bin';
.shell rm -f qjs_fileio.txt qjs_fileio.bin
"""),
("completion_vtab", """
CREATE TABLE alpha(a); CREATE TABLE alphabet(b);
SELECT DISTINCT candidate FROM completion('SELECT * FROM alp', 17) ORDER BY 1;
SELECT DISTINCT candidate FROM completion('sel', 3) ORDER BY 1 LIMIT 5;
SELECT DISTINCT candidate FROM completion('', 0) ORDER BY 1 LIMIT 8;
"""),
("builtin_vtabs", """
CREATE TABLE t(a PRIMARY KEY, b); CREATE INDEX tb ON t(b); INSERT INTO t SELECT value, value*2 FROM generate_series(1, 200);
SELECT name, path, pageno IS NOT NULL, ncell, payload FROM dbstat WHERE name = 't' ORDER BY path LIMIT 5;
SELECT name, sum(ncell), sum(payload) > 0 FROM dbstat GROUP BY name ORDER BY name;
"""),
("shell_dot_dump_restore", """
CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c REAL);
CREATE INDEX tb ON t(b); CREATE VIEW v AS SELECT b FROM t; CREATE TRIGGER tr AFTER INSERT ON t BEGIN SELECT 1; END;
INSERT INTO t VALUES(1,'x',1.5),(2,'it''s',NULL),(3,char(10),-0.0);
.output test_dump.sql
.dump
.output stdout
DROP VIEW v; DROP TABLE t;
.read test_dump.sql
SELECT * FROM t; SELECT * FROM v;
.shell rm -f test_dump.sql
"""),
("shell_modes_and_formats", """
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
"""),
("shell_import_export_blob", """
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
"""),
("shell_sha3sum_selftest", """
CREATE TABLE t(a, b); INSERT INTO t VALUES(1,'a'),(2,'b');
.sha3sum
.sha3sum --schema
.sha3sum --sha3-224
.schema
.fullschema
.indexes
.tables
"""),
]

SCRIPTS_EXT = {}

SCRIPTS_EXT["fts34"] = [
("fts3_basic", """
CREATE VIRTUAL TABLE d USING fts3(title, body);
INSERT INTO d VALUES('One','the quick brown fox'),('Two','jumps over the lazy dog'),('Three','a fox and a dog');
SELECT rowid, title FROM d WHERE d MATCH 'fox' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'fox AND dog' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'fox OR lazy' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH '"brown fox"' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'title:two' ORDER BY rowid;
SELECT snippet(d, '[', ']'), offsets(d) FROM d WHERE d MATCH 'dog' ORDER BY rowid;
"""),
("fts4_matchinfo_options", """
CREATE VIRTUAL TABLE d USING fts4(a, b, prefix='2,4', tokenize=porter, order=desc);
INSERT INTO d VALUES('running quickly','dogs running'),('runs and runners','a dog'),('quick brown','nothing here');
SELECT rowid FROM d WHERE d MATCH 'run*' ORDER BY rowid;
SELECT rowid, hex(matchinfo(d)) FROM d WHERE d MATCH 'dog' ORDER BY rowid;
SELECT rowid, hex(matchinfo(d, 'pcnalx')) FROM d WHERE d MATCH 'quick OR dogs' ORDER BY rowid;
SELECT docid FROM d WHERE d MATCH 'running NEAR dogs' ORDER BY docid;
INSERT INTO d(d) VALUES('optimize');
INSERT INTO d(d) VALUES('integrity-check');
"""),
("fts4_content_notindexed_languageid", """
CREATE TABLE src(id INTEGER PRIMARY KEY, t, u);
INSERT INTO src VALUES(1,'alpha beta','x'),(2,'beta gamma','y'),(3,'gamma delta','z');
CREATE VIRTUAL TABLE i USING fts4(content='src', t, notindexed=u);
INSERT INTO i(i) VALUES('rebuild');
SELECT docid FROM i WHERE i MATCH 'beta' ORDER BY docid;
SELECT docid FROM i WHERE i MATCH 'x';
CREATE VIRTUAL TABLE l USING fts4(a, languageid='lid');
INSERT INTO l(docid, a, lid) VALUES(1, 'hello', 0),(2, 'hello', 1),(3, 'world', 1);
SELECT docid FROM l WHERE l MATCH 'hello' AND lid = 1;
SELECT docid FROM l WHERE l MATCH 'hello' AND lid = 0;
"""),
("fts3_tokenizers_unicode", """
CREATE VIRTUAL TABLE s USING fts3(t, tokenize=simple);
CREATE VIRTUAL TABLE u USING fts3(t, tokenize=unicode61 'remove_diacritics=2' 'tokenchars=-');
INSERT INTO s VALUES('Hello World'),('fox-trot'),('naïve café');
INSERT INTO u VALUES('Hello World'),('fox-trot'),('naïve café');
SELECT rowid FROM s WHERE s MATCH 'hello';
SELECT rowid FROM s WHERE s MATCH 'fox';
SELECT rowid FROM u WHERE u MATCH 'fox-trot';
SELECT rowid FROM u WHERE u MATCH 'cafe';
SELECT rowid FROM u WHERE u MATCH 'naive';
"""),
("fts5_extras", """
CREATE VIRTUAL TABLE d USING fts5(a, b UNINDEXED, c, columnsize=0, detail=column);
INSERT INTO d VALUES('red green blue','keep','red car'),('green yellow','keep2','blue sky'),('blue blue blue','k3','nothing');
SELECT rowid FROM d WHERE d MATCH 'blue' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'a:red' ORDER BY rowid;
SELECT rowid, highlight(d, 0, '<', '>') FROM d WHERE d MATCH 'blue' ORDER BY rowid;
SELECT rowid FROM d WHERE d MATCH 'blue' ORDER BY bm25(d, 10.0, 0.0, 1.0) LIMIT 2;
INSERT INTO d(d, rank) VALUES('rank', 'bm25(2.0)');
INSERT INTO d(d) VALUES('optimize');
SELECT rowid FROM d WHERE d MATCH 'red OR yellow' ORDER BY rank;
"""),
("geopoly_basic", """
CREATE VIRTUAL TABLE gp USING geopoly(name);
INSERT INTO gp(_shape, name) VALUES('[[0,0],[10,0],[10,10],[0,10],[0,0]]', 'square'),('[[5,5],[15,5],[10,15],[5,5]]', 'triangle'),('[[20,20],[30,20],[30,30],[20,30]]', 'far');
SELECT name, geopoly_area(_shape) FROM gp ORDER BY name;
SELECT name FROM gp WHERE geopoly_overlap(_shape, '[[8,8],[12,8],[12,12],[8,12]]') ORDER BY name;
SELECT name FROM gp WHERE geopoly_within(_shape, '[[-1,-1],[11,-1],[11,11],[-1,11]]') ORDER BY name;
SELECT name FROM gp WHERE geopoly_contains_point(_shape, 7, 7) ORDER BY name;
SELECT geopoly_json('[[0,0],[1,0],[1,1]]'), geopoly_bbox('[[0,0],[3,0],[3,4],[0,4]]') IS NOT NULL, geopoly_ccw('[[0,0],[0,1],[1,1],[1,0]]') IS NOT NULL;
SELECT geopoly_regular(0,0,5,6) IS NOT NULL, round(geopoly_area(geopoly_regular(0,0,5,6)), 4), geopoly_svg('[[0,0],[1,0],[1,1]]') IS NOT NULL;
SELECT geopoly_xform('[[0,0],[1,0],[1,1]]', 1, 0, 0, 1, 5, 5), geopoly_group_bbox(_shape) IS NOT NULL FROM gp;
"""),
("stat4_planning", """
CREATE TABLE t(a, b, c);
CREATE INDEX tab ON t(a, b);
CREATE INDEX tc ON t(c);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<3000)
INSERT INTO t SELECT i%100, i%7, CASE WHEN i%1000=0 THEN 'rare' ELSE 'common' END FROM n;
ANALYZE;
SELECT idx, count(*) FROM sqlite_stat4 GROUP BY idx ORDER BY idx;
EXPLAIN QUERY PLAN SELECT * FROM t WHERE c = 'rare';
EXPLAIN QUERY PLAN SELECT * FROM t WHERE c = 'common' AND a = 5;
EXPLAIN QUERY PLAN SELECT * FROM t WHERE a = 50 AND b > 2;
SELECT count(*) FROM t WHERE c = 'rare';
SELECT count(*) FROM t WHERE a = 50 AND b > 2;
SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY 1, 2;
"""),
("bytecode_dbpage_vtabs", """
CREATE TABLE t(a PRIMARY KEY, b); INSERT INTO t SELECT value, value*2 FROM generate_series(1, 50);
SELECT opcode FROM bytecode('SELECT a + 1 FROM t WHERE b > 3') ORDER BY addr LIMIT 6;
SELECT DISTINCT type, name FROM tables_used('SELECT * FROM t WHERE a IN (SELECT b FROM t)') ORDER BY 1, 2;
SELECT pgno, length(data) FROM sqlite_dbpage WHERE pgno <= 2 ORDER BY pgno;
SELECT sql, ncol FROM sqlite_stmt WHERE sql LIKE 'SELECT%' LIMIT 1;
"""),
("session_free_misc_ext", """
SELECT sqlite_compileoption_used('ENABLE_FTS3'), sqlite_compileoption_used('ENABLE_FTS4'), sqlite_compileoption_used('ENABLE_GEOPOLY'), sqlite_compileoption_used('ENABLE_STAT4'), sqlite_compileoption_used('THREADSAFE=1');
SELECT sqlite_compileoption_get(0) IS NOT NULL, sqlite_threadsafe() IS NULL;
PRAGMA threads;
PRAGMA threads = 2;
PRAGMA threads;
CREATE TABLE t(a); WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<20000) INSERT INTO t SELECT (i*7919)%10007 FROM n;
SELECT count(*), sum(a), min(a), max(a) FROM (SELECT a FROM t ORDER BY a);
CREATE INDEX ta ON t(a); PRAGMA integrity_check;
"""),
]
