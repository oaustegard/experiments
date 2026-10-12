CREATE VIRTUAL TABLE r USING rtree_i32(id, a0, a1, b0, b1);
INSERT INTO r VALUES(1, 1.9, 4.2, 10, 20),(2, -5, 0, 1, 1);
SELECT * FROM r ORDER BY id;
CREATE VIRTUAL TABLE r2 USING rtree(id, x0, x1, +name TEXT);
INSERT INTO r2 VALUES(1, 0, 1, 'unit'),(2, 5, 6, 'far');
SELECT id, name FROM r2 WHERE x0 < 2 AND x1 > 0.5;
