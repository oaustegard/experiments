CREATE TABLE a(id, x); CREATE TABLE b(id, y);
INSERT INTO a VALUES(1,'a1'),(2,'a2'); INSERT INTO b VALUES(2,'b2'),(3,'b3');
SELECT a.id, b.id, x, y FROM a RIGHT JOIN b ON a.id=b.id ORDER BY coalesce(a.id,b.id);
SELECT a.id, b.id, x, y FROM a FULL OUTER JOIN b ON a.id=b.id ORDER BY coalesce(a.id,b.id);
