WITH RECURSIVE n(i, h) AS (SELECT 0, 5381 UNION ALL SELECT i+1, ((h << 5) + h + (i % 251)) & 4294967295 FROM n WHERE i < 5000)
SELECT h FROM n WHERE i = 5000;
SELECT hex(CAST(x'00' || printf('%c', 200) AS BLOB));
SELECT sum(unicode(substr('hello world', value, 1))) FROM json_each('[1,2,3,4,5,6,7,8,9,10,11]');
