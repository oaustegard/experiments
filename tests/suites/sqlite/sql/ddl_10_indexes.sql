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
