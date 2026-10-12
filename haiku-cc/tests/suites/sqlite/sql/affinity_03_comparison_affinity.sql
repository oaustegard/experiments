CREATE TABLE t(i INTEGER, t TEXT, x);
INSERT INTO t VALUES(5,'5',5);
SELECT i=5, i='5', t=5, t='5', x=5, x='5', i=t, i<'6', t<6, x<'6' FROM t;
SELECT 1='1', 1.0='1', '1'=1, 'a'<1, 1<'a', NULL<1, x'00'>'z', x'00'>1 FROM t;
SELECT 2 < 10, '2' < '10', 2 < '10', '2' < 10, CAST(2 AS TEXT) < CAST(10 AS TEXT);
