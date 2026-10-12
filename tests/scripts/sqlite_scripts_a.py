# SQL test scripts, part A (DDL, DML, select, joins, aggregates, windows, CTEs).
# Each entry: (slug, sql).  tests/scripts/gen_sqlite_scripts.py writes them to
# tests/suites/sqlite/sql/<category>_<nn>_<slug>.sql.  Scripts only use deterministic functions.

PEOPLE = """
CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL, boss INTEGER, hired TEXT);
INSERT INTO emp VALUES
 (1,'Ada','eng',120.5,NULL,'2015-03-01'),(2,'Bob','eng',100,1,'2016-07-15'),(3,'Cy','ops',80,1,'2017-01-20'),
 (4,'Di','ops',85.25,3,'2018-11-30'),(5,'Ed','eng',95,2,'2019-05-05'),(6,'Flo','hr',70,1,'2020-02-29'),
 (7,'Gus','hr',72,6,'2021-08-08'),(8,'Hal','eng',130,1,'2014-12-12'),(9,'Ivy','ops',NULL,3,'2022-04-04');
"""

SCRIPTS = {}

SCRIPTS["ddl"] = [
("create_table_basic", """
CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT NOT NULL, c REAL DEFAULT 1.5, d BLOB);
.schema t
INSERT INTO t(b) VALUES('x');
SELECT * FROM t;
"""),
("create_if_not_exists", """
CREATE TABLE t(a);
CREATE TABLE IF NOT EXISTS t(a, b);
CREATE TABLE t(a);
.tables
"""),
("alter_rename", """
CREATE TABLE t(a, b);
INSERT INTO t VALUES(1,2);
ALTER TABLE t RENAME TO u;
ALTER TABLE u RENAME COLUMN b TO c;
ALTER TABLE u ADD COLUMN d DEFAULT 'z';
SELECT * FROM u;
.schema
"""),
("alter_drop_column", """
CREATE TABLE t(a, b, c);
INSERT INTO t VALUES(1,2,3);
ALTER TABLE t DROP COLUMN b;
SELECT * FROM t;
ALTER TABLE t DROP COLUMN a;
SELECT * FROM t;
"""),
("drop_table", """
CREATE TABLE a(x); CREATE TABLE b(y);
DROP TABLE a;
DROP TABLE IF EXISTS nothere;
DROP TABLE nothere;
.tables
"""),
("without_rowid", """
CREATE TABLE kv(k TEXT PRIMARY KEY, v) WITHOUT ROWID;
INSERT INTO kv VALUES('b',2),('a',1),('c',3);
SELECT * FROM kv;
INSERT OR REPLACE INTO kv VALUES('a',10);
SELECT * FROM kv ORDER BY k DESC;
"""),
("constraints_check", """
CREATE TABLE t(a INTEGER CHECK(a > 0), b TEXT CHECK(length(b) < 4));
INSERT INTO t VALUES(1,'abc');
INSERT INTO t VALUES(0,'abc');
INSERT INTO t VALUES(1,'abcd');
SELECT count(*) FROM t;
"""),
("constraints_unique_notnull", """
CREATE TABLE t(a UNIQUE, b NOT NULL, c, UNIQUE(b,c));
INSERT INTO t VALUES(1,1,1);
INSERT INTO t VALUES(1,2,2);
INSERT INTO t VALUES(2,NULL,2);
INSERT INTO t VALUES(2,1,1);
INSERT INTO t VALUES(NULL,3,3);
INSERT INTO t VALUES(NULL,4,4);
SELECT * FROM t;
"""),
("foreign_keys", """
PRAGMA foreign_keys=ON;
CREATE TABLE p(id INTEGER PRIMARY KEY);
CREATE TABLE c(id INTEGER PRIMARY KEY, p REFERENCES p(id) ON DELETE CASCADE);
INSERT INTO p VALUES(1),(2);
INSERT INTO c VALUES(10,1),(11,2);
INSERT INTO c VALUES(12,3);
DELETE FROM p WHERE id=1;
SELECT * FROM c;
PRAGMA foreign_key_check;
"""),
("indexes", """
CREATE TABLE t(a,b,c);
CREATE INDEX i1 ON t(a);
CREATE UNIQUE INDEX i2 ON t(b,c);
CREATE INDEX i3 ON t(a) WHERE c > 5;
CREATE INDEX i4 ON t(lower(b));
INSERT INTO t VALUES(1,'A',1),(2,'B',7),(3,'C',9);
.indexes t
SELECT name, sql FROM sqlite_master WHERE type='index' ORDER BY name;
SELECT * FROM t WHERE lower(b)='b';
DROP INDEX i1;
.indexes t
"""),
("views", """
CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,'x'),(2,'y'),(3,'z');
CREATE VIEW v AS SELECT a*2 AS d, b FROM t WHERE a>1;
SELECT * FROM v;
CREATE VIEW w(p,q) AS SELECT d, upper(b) FROM v;
SELECT * FROM w ORDER BY p DESC;
DROP VIEW v;
SELECT * FROM w;
"""),
("temp_tables", """
CREATE TEMP TABLE t(a);
INSERT INTO t VALUES(1),(2);
CREATE TABLE main.m(a);
SELECT name FROM sqlite_temp_master;
SELECT sum(a) FROM temp.t;
.tables
"""),
("generated_columns", """
CREATE TABLE t(a INTEGER, b INTEGER AS (a*2) VIRTUAL, c TEXT GENERATED ALWAYS AS (a||'!') STORED);
INSERT INTO t(a) VALUES(1),(2),(3);
SELECT * FROM t;
UPDATE t SET a=a+10 WHERE a=2;
SELECT * FROM t ORDER BY a;
PRAGMA table_xinfo(t);
"""),
("strict_tables", """
CREATE TABLE t(a INTEGER, b TEXT, c REAL, d ANY) STRICT;
INSERT INTO t VALUES(1,'x',1.5,'any');
INSERT INTO t VALUES('abc','x',1.5,2);
INSERT INTO t VALUES(1,2,'3.5',x'00');
SELECT *, typeof(a), typeof(b), typeof(c) FROM t;
"""),
("table_info_pragmas", """
CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT DEFAULT 'q' NOT NULL, c);
CREATE INDEX ic ON t(c);
PRAGMA table_info(t);
PRAGMA index_list(t);
PRAGMA index_info(ic);
"""),
]

SCRIPTS["dml"] = [
("insert_multi", """
CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,'a'),(2,'b'),(3,'c');
INSERT INTO t SELECT a+10, b||b FROM t;
SELECT * FROM t;
SELECT changes(), total_changes();
"""),
("insert_default_values", """
CREATE TABLE t(id INTEGER PRIMARY KEY, x DEFAULT 'd', y DEFAULT (1+2));
INSERT INTO t DEFAULT VALUES;
INSERT INTO t DEFAULT VALUES;
SELECT * FROM t;
SELECT last_insert_rowid();
"""),
("update_basic", """
CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,10),(2,20),(3,30);
UPDATE t SET b=b+1 WHERE a>1;
UPDATE t SET a=a*100, b=NULL WHERE a=1;
SELECT * FROM t ORDER BY a;
SELECT changes();
"""),
("update_from", """
CREATE TABLE t(id,v); CREATE TABLE u(id,n);
INSERT INTO t VALUES(1,0),(2,0),(3,0);
INSERT INTO u VALUES(1,5),(3,7);
UPDATE t SET v=u.n FROM u WHERE t.id=u.id;
SELECT * FROM t;
"""),
("delete_basic", """
CREATE TABLE t(a);
INSERT INTO t VALUES(1),(2),(3),(4),(5),(6);
DELETE FROM t WHERE a%2=0;
SELECT * FROM t;
DELETE FROM t;
SELECT count(*) FROM t;
"""),
("upsert", """
CREATE TABLE t(k PRIMARY KEY, n);
INSERT INTO t VALUES('a',1);
INSERT INTO t VALUES('a',5) ON CONFLICT(k) DO UPDATE SET n=n+excluded.n;
INSERT INTO t VALUES('b',2) ON CONFLICT(k) DO NOTHING;
INSERT INTO t VALUES('b',9) ON CONFLICT(k) DO NOTHING;
SELECT * FROM t ORDER BY k;
"""),
("insert_or_variants", """
CREATE TABLE t(a UNIQUE, b);
INSERT INTO t VALUES(1,'one');
INSERT OR IGNORE INTO t VALUES(1,'uno');
INSERT OR REPLACE INTO t VALUES(1,'eins');
INSERT OR ABORT INTO t VALUES(1,'x');
INSERT OR FAIL INTO t VALUES(2,'two'),(1,'dup'),(3,'three');
REPLACE INTO t VALUES(4,'four');
SELECT * FROM t ORDER BY a;
"""),
("returning", """
CREATE TABLE t(id INTEGER PRIMARY KEY, v);
INSERT INTO t(v) VALUES('a'),('b') RETURNING id, upper(v);
UPDATE t SET v=v||'!' RETURNING *;
DELETE FROM t WHERE id=1 RETURNING v;
"""),
("rowid_alias", """
CREATE TABLE t(a);
INSERT INTO t VALUES('x'),('y');
SELECT rowid, oid, _rowid_, a FROM t;
INSERT INTO t(rowid, a) VALUES(100,'z');
INSERT INTO t VALUES('w');
SELECT rowid, a FROM t;
"""),
("autoincrement", """
CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v);
INSERT INTO t(v) VALUES('a'),('b'),('c');
DELETE FROM t WHERE id=3;
INSERT INTO t(v) VALUES('d');
SELECT * FROM t;
SELECT * FROM sqlite_sequence;
"""),
("triggers_after", """
CREATE TABLE t(a); CREATE TABLE log(msg);
CREATE TRIGGER ti AFTER INSERT ON t BEGIN INSERT INTO log VALUES('ins '||new.a); END;
CREATE TRIGGER tu AFTER UPDATE ON t BEGIN INSERT INTO log VALUES('upd '||old.a||'->'||new.a); END;
CREATE TRIGGER td AFTER DELETE ON t BEGIN INSERT INTO log VALUES('del '||old.a); END;
INSERT INTO t VALUES(1),(2);
UPDATE t SET a=a*10;
DELETE FROM t WHERE a=10;
SELECT * FROM log;
"""),
("triggers_instead_of", """
CREATE TABLE t(a,b);
CREATE VIEW v AS SELECT a, b FROM t;
CREATE TRIGGER vi INSTEAD OF INSERT ON v BEGIN INSERT INTO t VALUES(new.a*2, new.b); END;
CREATE TRIGGER vd INSTEAD OF DELETE ON v BEGIN DELETE FROM t WHERE a=old.a; END;
INSERT INTO v VALUES(1,'x'),(2,'y');
SELECT * FROM t;
DELETE FROM v WHERE a=2;
SELECT * FROM t;
"""),
("triggers_when_raise", """
CREATE TABLE t(a);
CREATE TRIGGER chk BEFORE INSERT ON t WHEN new.a < 0 BEGIN SELECT RAISE(ABORT, 'negative not allowed'); END;
INSERT INTO t VALUES(1);
INSERT INTO t VALUES(-1);
INSERT INTO t VALUES(2);
SELECT * FROM t;
"""),
]

SCRIPTS["select"] = [
("where_ops", PEOPLE + """
SELECT name FROM emp WHERE salary > 90 AND dept <> 'hr' ORDER BY name;
SELECT name FROM emp WHERE dept IN ('hr','ops') AND salary IS NOT NULL ORDER BY salary DESC;
SELECT name FROM emp WHERE salary BETWEEN 80 AND 100 ORDER BY id;
SELECT name FROM emp WHERE boss IS NULL OR salary IS NULL;
"""),
("like_glob", PEOPLE + """
SELECT name FROM emp WHERE name LIKE '_d%' ORDER BY id;
SELECT name FROM emp WHERE name LIKE '%Y' ORDER BY id;
SELECT name FROM emp WHERE name GLOB '[A-D]*' ORDER BY id;
SELECT 'a%b' LIKE 'a\\%b' ESCAPE '\\', 'axb' LIKE 'a\\%b' ESCAPE '\\';
"""),
("order_limit_offset", PEOPLE + """
SELECT name, salary FROM emp ORDER BY salary DESC NULLS LAST LIMIT 3;
SELECT name FROM emp ORDER BY id LIMIT 3 OFFSET 4;
SELECT name FROM emp ORDER BY dept, salary DESC, name LIMIT 5;
SELECT id FROM emp ORDER BY id LIMIT -1 OFFSET 7;
"""),
("distinct_group", PEOPLE + """
SELECT DISTINCT dept FROM emp ORDER BY dept;
SELECT dept, count(*) FROM emp GROUP BY dept ORDER BY 2 DESC, 1;
SELECT count(DISTINCT dept), count(salary), count(*) FROM emp;
"""),
("case_expr", PEOPLE + """
SELECT name, CASE WHEN salary IS NULL THEN 'unknown' WHEN salary>=100 THEN 'high' ELSE 'low' END FROM emp ORDER BY id;
SELECT name, CASE dept WHEN 'eng' THEN 1 WHEN 'ops' THEN 2 ELSE 3 END AS d FROM emp ORDER BY d, name;
"""),
("null_semantics", """
SELECT NULL=NULL, NULL IS NULL, NULL IS NOT NULL, 1 IS NULL, NULL AND 0, NULL OR 1, NULL AND 1, NOT NULL;
SELECT 1 IS 1, NULL IS NULL, 1 IS NOT 2, NULL IS NOT NULL;
SELECT coalesce(NULL,NULL,3), ifnull(NULL,'x'), nullif(1,1), nullif(1,2), iif(1>2,'a','b');
SELECT max(1,NULL,3), min(1,NULL,3), 5 IN (1,NULL), 5 NOT IN (1,NULL), 1 IN (1,NULL);
"""),
("expressions", """
SELECT 1+2*3, (1+2)*3, 7/2, 7.0/2, -7/2, 7%3, -7%3, 2*-3, 1<<3, 256>>4, 5&3, 5|3, ~5;
SELECT 1=1, 1==1, 1!=2, 1<>2, 'a'<'b', 'a'||'b'||1||2.5, -(-3), +'4';
SELECT 9223372036854775807 + 1, 9223372036854775807 * 2, -9223372036854775808 - 1;
SELECT 1/0, 1%0, 1.0/0, 0.0/0;
"""),
("subquery_scalar_exists", PEOPLE + """
SELECT name, (SELECT count(*) FROM emp e2 WHERE e2.boss=emp.id) AS reports FROM emp ORDER BY reports DESC, id LIMIT 4;
SELECT name FROM emp WHERE EXISTS (SELECT 1 FROM emp e2 WHERE e2.boss=emp.id) ORDER BY id;
SELECT name FROM emp WHERE salary > (SELECT avg(salary) FROM emp) ORDER BY id;
SELECT name FROM emp WHERE id IN (SELECT boss FROM emp) ORDER BY id;
"""),
("compound_select", """
CREATE TABLE a(x); CREATE TABLE b(x);
INSERT INTO a VALUES(1),(2),(3),(3); INSERT INTO b VALUES(3),(4),(2),(2);
SELECT x FROM a UNION SELECT x FROM b ORDER BY 1;
SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY 1;
SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY 1;
SELECT x FROM a EXCEPT SELECT x FROM b ORDER BY 1;
"""),
("values_clause", """
VALUES(1,'a'),(2,'b'),(3,'c');
SELECT * FROM (SELECT column1 AS x, column2 AS y FROM (VALUES(1,2),(3,4))) WHERE x>1;
SELECT column1+column2 FROM (VALUES(1,2),(3,4));
"""),
("derived_tables", PEOPLE + """
SELECT d, n FROM (SELECT dept AS d, count(*) AS n FROM emp GROUP BY dept) WHERE n>2 ORDER BY d;
SELECT a.name, b.name FROM (SELECT * FROM emp WHERE id<3) a, (SELECT * FROM emp WHERE id>7) b ORDER BY 1,2;
"""),
("output_modes_column", PEOPLE + """
.headers on
.mode column
SELECT id, name, dept, salary FROM emp WHERE id<5;
.mode line
SELECT id, name FROM emp WHERE id=2;
"""),
("output_modes_box_json", PEOPLE + """
.mode box
SELECT id, name, salary FROM emp WHERE id<4;
.mode json
SELECT id, name, salary FROM emp WHERE id<4;
.mode csv
.headers on
SELECT * FROM emp WHERE id<3;
.mode markdown
SELECT name, dept FROM emp WHERE id<3;
"""),
("output_modes_insert_tabs", PEOPLE + """
.mode insert emp2
SELECT id, name FROM emp WHERE id<3;
.mode tabs
SELECT id, name FROM emp WHERE id<3;
.mode quote
SELECT id, name, salary, hired FROM emp WHERE id IN (1,9);
.nullvalue <null>
.mode list
SELECT salary, boss FROM emp WHERE id IN (1,9);
"""),
]

SCRIPTS["join"] = [
("inner_join", """
CREATE TABLE a(id, x); CREATE TABLE b(id, y);
INSERT INTO a VALUES(1,'a1'),(2,'a2'),(3,'a3'); INSERT INTO b VALUES(2,'b2'),(3,'b3'),(4,'b4'),(3,'b3b');
SELECT a.id, x, y FROM a JOIN b ON a.id=b.id ORDER BY 1,3;
SELECT a.id, x, y FROM a, b WHERE a.id=b.id ORDER BY 1,3;
"""),
("left_join", """
CREATE TABLE a(id, x); CREATE TABLE b(id, y);
INSERT INTO a VALUES(1,'a1'),(2,'a2'),(3,'a3'); INSERT INTO b VALUES(2,'b2'),(3,'b3'),(4,'b4');
SELECT a.id, x, y FROM a LEFT JOIN b ON a.id=b.id ORDER BY 1;
SELECT a.id FROM a LEFT JOIN b ON a.id=b.id WHERE b.id IS NULL;
"""),
("right_full_join", """
CREATE TABLE a(id, x); CREATE TABLE b(id, y);
INSERT INTO a VALUES(1,'a1'),(2,'a2'); INSERT INTO b VALUES(2,'b2'),(3,'b3');
SELECT a.id, b.id, x, y FROM a RIGHT JOIN b ON a.id=b.id ORDER BY coalesce(a.id,b.id);
SELECT a.id, b.id, x, y FROM a FULL OUTER JOIN b ON a.id=b.id ORDER BY coalesce(a.id,b.id);
"""),
("cross_join", """
CREATE TABLE a(x); CREATE TABLE b(y);
INSERT INTO a VALUES(1),(2),(3); INSERT INTO b VALUES('p'),('q');
SELECT x,y FROM a CROSS JOIN b ORDER BY x,y;
SELECT count(*) FROM a, a AS a2, a AS a3;
"""),
("natural_using", """
CREATE TABLE a(id, v); CREATE TABLE b(id, w);
INSERT INTO a VALUES(1,10),(2,20); INSERT INTO b VALUES(2,200),(3,300);
SELECT * FROM a NATURAL JOIN b;
SELECT * FROM a JOIN b USING(id);
SELECT * FROM a NATURAL LEFT JOIN b ORDER BY id;
"""),
("self_join", PEOPLE + """
SELECT e.name, b.name AS boss FROM emp e LEFT JOIN emp b ON e.boss=b.id ORDER BY e.id;
SELECT a.name, b.name FROM emp a JOIN emp b ON a.dept=b.dept AND a.id<b.id ORDER BY 1,2 LIMIT 8;
"""),
("three_way_join", """
CREATE TABLE cust(id, name); CREATE TABLE ord(id, cust, item); CREATE TABLE item(id, label, price);
INSERT INTO cust VALUES(1,'ann'),(2,'ben');
INSERT INTO item VALUES(1,'pen',1.5),(2,'ink',4),(3,'pad',2.25);
INSERT INTO ord VALUES(1,1,1),(2,1,3),(3,2,2),(4,2,2),(5,1,2);
SELECT c.name, i.label, i.price FROM cust c JOIN ord o ON o.cust=c.id JOIN item i ON i.id=o.item ORDER BY c.name, o.id;
SELECT c.name, sum(i.price) FROM cust c JOIN ord o ON o.cust=c.id JOIN item i ON i.id=o.item GROUP BY c.name ORDER BY 1;
"""),
("join_with_index", """
CREATE TABLE a(id INTEGER PRIMARY KEY, k); CREATE TABLE b(id INTEGER PRIMARY KEY, k);
CREATE INDEX bk ON b(k);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<200)
INSERT INTO a SELECT i, i%17 FROM n;
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<200)
INSERT INTO b SELECT i, i%13 FROM n;
SELECT count(*), sum(a.id), sum(b.id) FROM a JOIN b ON a.k=b.k;
"""),
("lateral_like_correlated", PEOPLE + """
SELECT name, (SELECT max(salary) FROM emp e2 WHERE e2.dept=e.dept) AS dmax FROM emp e ORDER BY id;
SELECT e.name FROM emp e WHERE salary = (SELECT max(salary) FROM emp WHERE dept=e.dept) ORDER BY 1;
"""),
("join_on_expression", """
CREATE TABLE a(x); CREATE TABLE b(y);
INSERT INTO a VALUES(1),(2),(3),(4); INSERT INTO b VALUES(2),(4),(6),(8);
SELECT x,y FROM a JOIN b ON y = x*2 ORDER BY x;
SELECT x,y FROM a JOIN b ON y BETWEEN x AND x+2 ORDER BY x,y;
"""),
]

SCRIPTS["aggregate"] = [
("basic_aggs", PEOPLE + """
SELECT count(*), count(salary), sum(salary), avg(salary), min(salary), max(salary), total(salary) FROM emp;
SELECT dept, sum(salary), avg(salary), min(name), max(name) FROM emp GROUP BY dept ORDER BY dept;
"""),
("group_concat", PEOPLE + """
SELECT dept, group_concat(name) FROM (SELECT * FROM emp ORDER BY id) GROUP BY dept ORDER BY dept;
SELECT group_concat(name, '|') FROM (SELECT name FROM emp ORDER BY name DESC);
SELECT dept, string_agg(name, ';') FROM (SELECT * FROM emp ORDER BY id) GROUP BY dept ORDER BY 1;
"""),
("having", PEOPLE + """
SELECT dept, count(*) c FROM emp GROUP BY dept HAVING c>2 ORDER BY dept;
SELECT dept, avg(salary) a FROM emp GROUP BY dept HAVING avg(salary) > 85 ORDER BY dept;
"""),
("agg_empty", """
CREATE TABLE t(a);
SELECT count(*), sum(a), total(a), avg(a), min(a), max(a), group_concat(a) FROM t;
SELECT a, count(*) FROM t GROUP BY a;
"""),
("agg_filter", PEOPLE + """
SELECT count(*) FILTER (WHERE dept='eng'), sum(salary) FILTER (WHERE salary>90), count(*) FROM emp;
SELECT dept, count(*) FILTER (WHERE salary IS NULL) FROM emp GROUP BY dept ORDER BY dept;
"""),
("agg_distinct", """
CREATE TABLE t(a,b);
INSERT INTO t VALUES(1,1),(1,2),(2,2),(2,2),(3,NULL);
SELECT count(DISTINCT a), count(DISTINCT b), sum(DISTINCT a), sum(DISTINCT b), group_concat(DISTINCT a) FROM t;
"""),
("agg_ordered_set", PEOPLE + """
SELECT group_concat(name ORDER BY salary DESC) FROM emp WHERE salary IS NOT NULL;
SELECT dept, group_concat(name, ',' ORDER BY name) FROM emp GROUP BY dept ORDER BY dept;
"""),
("group_by_expr", PEOPLE + """
SELECT substr(hired,1,4) AS yr, count(*) FROM emp GROUP BY yr ORDER BY yr;
SELECT salary>90, count(*) FROM emp GROUP BY 1 ORDER BY 1;
SELECT dept, boss IS NULL AS top, count(*) FROM emp GROUP BY dept, top ORDER BY 1,2;
"""),
("sum_overflow_types", """
CREATE TABLE t(a);
INSERT INTO t VALUES(9223372036854775807),(1);
SELECT total(a), typeof(total(a)) FROM t;
SELECT sum(a) FROM t;
"""),
("minmax_bare_columns", PEOPLE + """
SELECT name, max(salary) FROM emp;
SELECT dept, name, min(hired) FROM emp GROUP BY dept ORDER BY dept;
"""),
("agg_math_stats", """
CREATE TABLE t(x);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<100) INSERT INTO t SELECT i*1.5 FROM n;
SELECT count(*), sum(x), avg(x), min(x), max(x), sum(x*x), round(avg(x*x)-avg(x)*avg(x),4) FROM t;
"""),
]

SCRIPTS["window"] = [
("row_number_rank", PEOPLE + """
SELECT name, dept, salary, row_number() OVER (PARTITION BY dept ORDER BY salary DESC) rn,
 rank() OVER (ORDER BY dept) rk, dense_rank() OVER (ORDER BY dept) dr FROM emp WHERE salary IS NOT NULL ORDER BY dept, rn;
"""),
("running_sum", PEOPLE + """
SELECT id, salary, sum(salary) OVER (ORDER BY id) AS run FROM emp WHERE salary IS NOT NULL ORDER BY id;
SELECT id, salary, sum(salary) OVER (PARTITION BY dept ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM emp WHERE salary IS NOT NULL ORDER BY dept, id;
"""),
("lag_lead", PEOPLE + """
SELECT id, name, lag(name) OVER (ORDER BY id), lead(name, 2, 'none') OVER (ORDER BY id) FROM emp ORDER BY id;
SELECT id, salary, salary - lag(salary,1,0) OVER (PARTITION BY dept ORDER BY id) FROM emp ORDER BY id;
"""),
("first_last_nth", PEOPLE + """
SELECT id, dept, first_value(name) OVER w, last_value(name) OVER w, nth_value(name, 2) OVER w
FROM emp WINDOW w AS (PARTITION BY dept ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) ORDER BY id;
"""),
("ntile_percent", """
CREATE TABLE t(v); INSERT INTO t VALUES(10),(20),(20),(30),(40),(50),(60),(70);
SELECT v, ntile(3) OVER (ORDER BY v), percent_rank() OVER (ORDER BY v), cume_dist() OVER (ORDER BY v) FROM t;
"""),
("moving_average", """
CREATE TABLE t(d, v);
INSERT INTO t VALUES(1,10),(2,20),(3,30),(4,40),(5,50),(6,60);
SELECT d, v, avg(v) OVER (ORDER BY d ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t;
SELECT d, sum(v) OVER (ORDER BY d ROWS 2 PRECEDING) FROM t;
"""),
("range_groups_frames", """
CREATE TABLE t(g, v); INSERT INTO t VALUES(1,1),(1,2),(2,2),(2,3),(3,5),(5,8);
SELECT g, v, sum(v) OVER (ORDER BY g RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t;
SELECT g, v, count(*) OVER (ORDER BY g GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t;
SELECT g, v, sum(v) OVER (ORDER BY g ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW) FROM t;
"""),
("window_filter_agg", PEOPLE + """
SELECT name, count(*) FILTER (WHERE salary > 90) OVER (PARTITION BY dept) FROM emp ORDER BY id;
SELECT dept, name, salary, salary*1.0/sum(salary) OVER (PARTITION BY dept) AS share FROM emp WHERE salary IS NOT NULL ORDER BY dept, name;
"""),
("window_in_subquery", PEOPLE + """
SELECT name, salary FROM (SELECT name, salary, rank() OVER (PARTITION BY dept ORDER BY salary DESC) r FROM emp WHERE salary IS NOT NULL) WHERE r=1 ORDER BY name;
"""),
("window_named_inherit", """
CREATE TABLE t(a,b); INSERT INTO t VALUES(1,5),(1,6),(2,7),(2,8),(2,9);
SELECT a, b, sum(b) OVER (w ORDER BY b), count(*) OVER w FROM t WINDOW w AS (PARTITION BY a) ORDER BY a, b;
"""),
("window_string_agg", """
CREATE TABLE t(g,v); INSERT INTO t VALUES('a','x'),('a','y'),('b','z'),('a','w');
SELECT g, v, group_concat(v) OVER (PARTITION BY g ORDER BY rowid) FROM t ORDER BY rowid;
"""),
]

SCRIPTS["cte"] = [
("cte_basic", PEOPLE + """
WITH eng AS (SELECT * FROM emp WHERE dept='eng') SELECT name FROM eng WHERE salary>100 ORDER BY name;
WITH a(x) AS (SELECT 1), b(y) AS (SELECT x+1 FROM a) SELECT * FROM a, b;
"""),
("cte_recursive_count", """
WITH RECURSIVE c(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM c WHERE n<10) SELECT n, n*n FROM c;
WITH RECURSIVE c(n) AS (SELECT 1 UNION SELECT (n*2)%11 FROM c) SELECT group_concat(n) FROM c;
"""),
("cte_fib", """
WITH RECURSIVE fib(a,b) AS (SELECT 0,1 UNION ALL SELECT b,a+b FROM fib WHERE b<1000) SELECT group_concat(a) FROM fib;
WITH RECURSIVE f(n,fact) AS (SELECT 1,1 UNION ALL SELECT n+1,fact*(n+1) FROM f WHERE n<20) SELECT n,fact FROM f WHERE n IN (5,10,20);
"""),
("cte_tree", PEOPLE + """
WITH RECURSIVE tree(id,name,depth,path) AS (
 SELECT id,name,0,name FROM emp WHERE boss IS NULL
 UNION ALL SELECT e.id,e.name,t.depth+1,t.path||'/'||e.name FROM emp e JOIN tree t ON e.boss=t.id)
SELECT printf('%s%s', substr('          ',1,depth*2), name), path FROM tree ORDER BY path;
"""),
("cte_materialized", """
WITH t AS MATERIALIZED (SELECT 1 AS x UNION ALL SELECT 2), u AS NOT MATERIALIZED (SELECT x*10 AS y FROM t)
SELECT x, y FROM t, u ORDER BY 1,2;
"""),
("cte_in_dml", """
CREATE TABLE t(a);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<5) INSERT INTO t SELECT i FROM n;
WITH big AS (SELECT a FROM t WHERE a>3) DELETE FROM t WHERE a IN (SELECT a FROM big);
SELECT * FROM t;
WITH RECURSIVE s(x) AS (SELECT 10) UPDATE t SET a=a+(SELECT x FROM s);
SELECT * FROM t;
"""),
("cte_mandelbrot", """
WITH RECURSIVE xaxis(x) AS (VALUES(-2.0) UNION ALL SELECT x+0.25 FROM xaxis WHERE x<0.5),
 yaxis(y) AS (VALUES(-1.0) UNION ALL SELECT y+0.5 FROM yaxis WHERE y<1.0),
 m(iter,cx,cy,x,y) AS (SELECT 0,x,y,0.0,0.0 FROM xaxis,yaxis UNION ALL
   SELECT iter+1,cx,cy,x*x-y*y+cx,2.0*x*y+cy FROM m WHERE (x*x+y*y)<4.0 AND iter<28),
 m2(iter,cx,cy) AS (SELECT max(iter),cx,cy FROM m GROUP BY cx,cy),
 a(t) AS (SELECT group_concat(substr(' .+*#',1+min(iter/7,4),1),'') FROM m2 GROUP BY cy)
SELECT t FROM a;
"""),
("cte_permutations", """
WITH RECURSIVE digits(z,lp) AS (VALUES('1',1) UNION ALL SELECT CAST(lp+1 AS TEXT),lp+1 FROM digits WHERE lp<4)
SELECT group_concat(z,'') FROM digits;
WITH RECURSIVE perm(s,rest) AS (SELECT '', 'abc' UNION ALL SELECT s||substr(rest,i,1), substr(rest,1,i-1)||substr(rest,i+1)
 FROM perm, (SELECT 1 i UNION ALL SELECT 2 UNION ALL SELECT 3) WHERE i<=length(rest))
SELECT s FROM perm WHERE rest='' ORDER BY s;
"""),
("cte_series_limit", """
WITH RECURSIVE s(v) AS (SELECT 5 UNION ALL SELECT v+5 FROM s WHERE v<50) SELECT sum(v), count(*), avg(v) FROM s;
WITH RECURSIVE s(v) AS (SELECT 1 UNION ALL SELECT v*3 FROM s LIMIT 8) SELECT group_concat(v,' ') FROM s;
"""),
]

SCRIPTS["subquery"] = [
("in_not_in", """
CREATE TABLE a(x); CREATE TABLE b(y); INSERT INTO a VALUES(1),(2),(3),(NULL); INSERT INTO b VALUES(2),(3),(4);
SELECT x FROM a WHERE x IN (SELECT y FROM b);
SELECT x FROM a WHERE x NOT IN (SELECT y FROM b);
SELECT x, x IN (SELECT y FROM b) FROM a;
"""),
("row_values", """
CREATE TABLE t(a,b); INSERT INTO t VALUES(1,2),(3,4),(5,6);
SELECT * FROM t WHERE (a,b) = (3,4);
SELECT * FROM t WHERE (a,b) IN (VALUES(1,2),(5,6));
SELECT (1,2) < (1,3), (1,2) = (1,2), (2,1) > (1,9);
"""),
("exists_correlated", """
CREATE TABLE s(id,name); CREATE TABLE o(sid,amt);
INSERT INTO s VALUES(1,'a'),(2,'b'),(3,'c'); INSERT INTO o VALUES(1,10),(1,20),(3,5);
SELECT name FROM s WHERE EXISTS(SELECT 1 FROM o WHERE o.sid=s.id) ORDER BY 1;
SELECT name FROM s WHERE NOT EXISTS(SELECT 1 FROM o WHERE o.sid=s.id) ORDER BY 1;
SELECT name, (SELECT sum(amt) FROM o WHERE sid=s.id) FROM s ORDER BY 1;
"""),
("scalar_subquery_empty", """
SELECT (SELECT 1 WHERE 0), (SELECT 2), (SELECT 3 UNION SELECT 4);
SELECT typeof((SELECT NULL)), (SELECT count(*) FROM (SELECT 1 UNION ALL SELECT 2));
"""),
("compare_to_subquery", """
CREATE TABLE t(v); INSERT INTO t VALUES(3),(7),(11);
SELECT 5 > (SELECT min(v) FROM t), 5 > (SELECT max(v) FROM t);
SELECT v FROM t WHERE v > (SELECT avg(v) FROM t);
SELECT v FROM t WHERE v = (SELECT max(v) FROM t WHERE v < 10);
"""),
]
