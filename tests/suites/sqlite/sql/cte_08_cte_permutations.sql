WITH RECURSIVE digits(z,lp) AS (VALUES('1',1) UNION ALL SELECT CAST(lp+1 AS TEXT),lp+1 FROM digits WHERE lp<4)
SELECT group_concat(z,'') FROM digits;
WITH RECURSIVE perm(s,rest) AS (SELECT '', 'abc' UNION ALL SELECT s||substr(rest,i,1), substr(rest,1,i-1)||substr(rest,i+1)
 FROM perm, (SELECT 1 i UNION ALL SELECT 2 UNION ALL SELECT 3) WHERE i<=length(rest))
SELECT s FROM perm WHERE rest='' ORDER BY s;
