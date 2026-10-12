CREATE TABLE n(i);
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<300) INSERT INTO n SELECT i FROM s;
SELECT count(*), sum(a.i*b.i) FROM n a, n b WHERE a.i<=b.i AND (a.i+b.i)%7=0;
SELECT i%10 AS k, count(*), sum(i), min(i), max(i) FROM n GROUP BY k ORDER BY k;
CREATE INDEX ni ON n(i%10);
SELECT count(*) FROM n WHERE i%10=3;
