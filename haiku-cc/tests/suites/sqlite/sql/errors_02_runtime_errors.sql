CREATE TABLE t(a);
SELECT nosuchcol FROM t;
SELECT * FROM nosuchtable;
SELECT nosuchfunc(1);
SELECT abs();
SELECT abs(1,2);
INSERT INTO t VALUES(1,2);
SELECT 'ok';
