CREATE TABLE a(id, x); CREATE TABLE b(id, y);
INSERT INTO a VALUES(1,'a1'),(2,'a2'),(3,'a3'); INSERT INTO b VALUES(2,'b2'),(3,'b3'),(4,'b4');
SELECT a.id, x, y FROM a LEFT JOIN b ON a.id=b.id ORDER BY 1;
SELECT a.id FROM a LEFT JOIN b ON a.id=b.id WHERE b.id IS NULL;
