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
