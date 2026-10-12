CREATE VIRTUAL TABLE r USING rtree(id, minx, maxx, miny, maxy);
INSERT INTO r VALUES(1, 0,10, 0,10),(2, 5,15, 5,15),(3, 20,30, 20,30),(4, 8,9, 8,9);
SELECT id FROM r WHERE minx>=5 AND maxx<=15 ORDER BY id;
SELECT id FROM r WHERE maxx>=9 AND minx<=9 AND maxy>=9 AND miny<=9 ORDER BY id;
SELECT count(*) FROM r;
