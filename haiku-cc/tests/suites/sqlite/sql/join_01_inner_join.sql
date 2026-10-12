CREATE TABLE a(id, x); CREATE TABLE b(id, y);
INSERT INTO a VALUES(1,'a1'),(2,'a2'),(3,'a3'); INSERT INTO b VALUES(2,'b2'),(3,'b3'),(4,'b4'),(3,'b3b');
SELECT a.id, x, y FROM a JOIN b ON a.id=b.id ORDER BY 1,3;
SELECT a.id, x, y FROM a, b WHERE a.id=b.id ORDER BY 1,3;
