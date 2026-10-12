CREATE VIRTUAL TABLE r USING rtree(id, x0, x1);
INSERT INTO r SELECT value, value*10, value*10+5 FROM json_each('[1,2,3,4,5,6,7,8]');
DELETE FROM r WHERE id%2=0;
UPDATE r SET x1 = x1 + 100 WHERE id=3;
SELECT * FROM r ORDER BY id;
SELECT id FROM r WHERE x0 < 40 AND x1 > 30 ORDER BY id;
