CREATE TABLE t(a INTEGER, b TEXT, c REAL, d BLOB);
INSERT INTO t VALUES(1,'x''y',1.5,x'00ff'),(NULL,'multi
line',NULL,NULL);
CREATE INDEX ti ON t(a);
CREATE VIEW v AS SELECT a FROM t;
.dump
.schema --indent
